#src\langgraph_mcp_agent\api\server.py
"""FastAPI服务器"""
# [保留所有imports不变]
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import json
import time
import asyncio
from datetime import datetime
from ..main import create_agent, LangGraphMCPAgent
from ..core.session_manager import session_manager
from ..utils.logger import setup_logger
import structlog

setup_logger()
logger = structlog.get_logger()

agent: Optional[LangGraphMCPAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent
    try:
        agent = create_agent()
        await agent.initialize()
        logger.info("FastAPI服务器启动，Agent系统已初始化")
        yield
    finally:
        if agent:
            await agent.cleanup()
        logger.info("FastAPI服务器关闭，资源已清理")

app = FastAPI(
    title="LangGraph MCP Chat API",
    description="OpenAI兼容的多轮对话聊天API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: user/assistant/system")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="langgraph-mcp-agent", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="消息列表")
    stream: bool = Field(default=False, description="是否流式返回")
    temperature: Optional[float] = Field(default=0.7, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    session_id: Optional[str] = Field(default=None, description="会话ID")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "LangGraph MCP Chat API",
        "version": "2.0.0",
        "status": "running",
        "api": "/v1/chat/completions",
        "docs": "/docs"
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成API"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent未初始化")
    
    session_id = request.session_id or session_manager.create_session()
    user_message = request.messages[-1].content if request.messages else ""
    
    if not user_message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")
    
    logger.info(f"处理聊天请求 - 会话: {session_id}, 流式: {request.stream}")
    
    if not request.stream:
        return await handle_non_streaming(user_message, session_id, request)
    else:
        return StreamingResponse(
            handle_streaming(user_message, session_id, request),
            media_type="text/event-stream"
        )

async def handle_non_streaming(message: str, session_id: str, request: ChatCompletionRequest):
    """处理非流式请求 - 适配新的三步格式"""
    try:
        # 收集所有输出
        full_response = ""
        
        async for chunk in agent.process_message(message, session_id, stream=True):
            if chunk["type"] == "planning":
                full_response += f"【{chunk.get('title', '第一步：任务规划')}】\n{chunk['content']}\n\n"
            elif chunk["type"] == "expert_result":
                full_response += f"【{chunk.get('title', '第二步：专家执行')}】\n{chunk['content']}\n\n"
            elif chunk["type"] == "final_result":
                full_response += f"【{chunk.get('title', '第三步：最终答案')}】\n{chunk['content']}"
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{session_id}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_response),
                    finish_reason="stop"
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"处理非流式请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_streaming(message: str, session_id: str, request: ChatCompletionRequest):
    """处理流式请求 - 修复版本"""
    try:
        completion_id = f"chatcmpl-{session_id}"
        
        # 发送初始chunk
        start_chunk = {
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {'role': 'assistant', 'content': ''},
                'finish_reason': None
            }]
        }
        yield f"data: {json.dumps(start_chunk)}\n\n"
        
        # 处理消息并流式输出
        buffer = ""
        last_flush_time = time.time()
        
        async for chunk in agent.process_message(message, session_id, stream=True):
            if chunk["type"] == "stream_chunk":
                content = chunk["content"]
                buffer += content
                
                # 每隔0.5秒或缓冲区超过200字符就发送
                if time.time() - last_flush_time > 0.5 or len(buffer) > 200:
                    if buffer:
                        delta_chunk = {
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': request.model,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': buffer},
                                'finish_reason': None
                            }]
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"
                        buffer = ""
                        last_flush_time = time.time()
            
            elif chunk["type"] == "error":
                # 错误处理
                error_chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': request.model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': f"\n[错误] {chunk['content']}"},
                        'finish_reason': None
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # 发送剩余的缓冲区内容
        if buffer:
            final_chunk = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': request.model,
                'choices': [{
                    'index': 0,
                    'delta': {'content': buffer},
                    'finish_reason': None
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
        
        # 发送结束标记
        end_chunk = {
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"处理流式请求失败: {e}")
        error_msg = json.dumps({"error": str(e)})
        yield f"data: {error_msg}\n\n"

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "langgraph-mcp-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "langgraph"
            }
        ]
    }

@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_manager.delete_session(session_id):
        return {"status": "success", "message": f"会话 {session_id} 已删除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")

@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话信息"""
    session = session_manager.get_session(session_id)
    if session:
        return {
            "id": session["id"],
            "created_at": session["created_at"].isoformat(),
            "last_accessed": session["last_accessed"].isoformat(),
            "message_count": len(session["messages"])
        }
    else:
        raise HTTPException(status_code=404, detail="会话不存在")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def get_status():
    """获取系统状态"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent未初始化")
    
    status = await agent.get_status()
    return status

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """运行FastAPI服务器"""
    import uvicorn
    uvicorn.run(
        "langgraph_mcp_agent.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()