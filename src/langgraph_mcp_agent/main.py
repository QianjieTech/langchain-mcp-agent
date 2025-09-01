# src/langgraph_mcp_agent/main.py
"""LangGraph MCP Agent主模块 - 增强版（保留原有功能）"""
import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field
from .graph.state_graph import MultiAgentGraph
from .core.session_manager import session_manager
from .utils.logger import setup_logger
import structlog

setup_logger()
logger = structlog.get_logger()

@dataclass
class UnifiedOutput:
    """统一输出格式"""
    step: int
    step_name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class StreamingOutputCollector:
    """流式输出收集器 - 非阻塞版本"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.completed = asyncio.Event()
        self.step_count = 0
        self.expected_steps = 3
        
    async def add(self, output: UnifiedOutput):
        """添加输出到队列"""
        await self.queue.put(output)
        logger.debug(f"添加输出到队列 - 步骤{output.step}: {output.step_name}")
        
        # 检查是否完成
        if output.step == 3 or output.step == -1:  # 最后一步或错误
            self.completed.set()
    
    async def stream(self) -> AsyncGenerator[str, None]:
        """流式输出 - 真正的非阻塞流"""
        try:
            while True:
                # 使用非阻塞方式获取输出
                try:
                    # 等待最多0.1秒获取新输出
                    output = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=0.1
                    )
                    
                    # 格式化并输出
                    formatted = self.format_output(output)
                    yield formatted
                    
                except asyncio.TimeoutError:
                    # 没有新输出，检查是否完成
                    if self.completed.is_set() and self.queue.empty():
                        logger.info("流式输出完成")
                        break
                    # 继续等待
                    continue
                    
        except Exception as e:
            logger.error(f"流式输出错误: {e}")
            yield f"\n[错误] {str(e)}\n"
    
    def format_output(self, output: UnifiedOutput) -> str:
        """格式化单个输出"""
        formatted = f"\n【步骤{output.step}：{output.step_name}】\n"
        
        # 处理元数据
        if "mcp_tools" in output.metadata:
            tools = output.metadata["mcp_tools"]
            if tools:
                formatted += f"使用MCP工具: {', '.join(tools)}\n"
        
        if "model_used" in output.metadata:
            model_info = output.metadata["model_used"]
            if model_info:
                formatted += f"使用模型: {model_info.get('model_name', 'unknown')} - {model_info.get('description', '')}\n"
        
        formatted += f"{output.content}\n"
        return formatted

class LangGraphMCPAgent:
    """主Agent系统 - 增强版"""
    
    def __init__(self):
        self.graph = MultiAgentGraph()
        self.initialized = False
        
    async def initialize(self):
        """初始化系统"""
        if self.initialized:
            return
            
        try:
            await self.graph.coordination_agent.initialize_tools()
            self.initialized = True
            logger.info("LangGraph MCP Agent系统初始化完成")
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            self.initialized = True
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """处理用户消息 - 增强版"""
        
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 添加用户消息到会话
            session_manager.add_message(session_id, "user", message)
            context = session_manager.get_context_for_agent(session_id)
            
            # 时间感知处理
            time_context = ""
            if self._needs_time_info(message):
                time_context = await self._get_time_from_mcp()
                logger.info(f"获取MCP时间上下文: {time_context}")
            
            # 构建增强查询
            enhanced_query = self._build_enhanced_query(message, context, time_context)
            
            if stream:
                # 创建输出收集器
                collector = StreamingOutputCollector()
                
                # 启动执行任务（后台运行）
                execution_task = asyncio.create_task(
                    self._execute_with_streaming(
                        enhanced_query, message, session_id, collector
                    )
                )
                
                # 立即开始流式输出
                async for chunk in collector.stream():
                    yield {
                        "type": "stream_chunk",
                        "content": chunk,
                        "timestamp": time.time()
                    }
                
                # 确保执行任务完成
                try:
                    await execution_task
                except Exception as e:
                    logger.error(f"执行任务错误: {e}")
                
                # 发送完成标记
                yield {
                    "type": "stream_complete",
                    "execution_time": time.time() - start_time
                }
                
            else:
                # 非流式：执行并等待完成
                collector = StreamingOutputCollector()
                await self._execute_with_streaming(
                    enhanced_query, message, session_id, collector
                )
                
                # 收集所有输出
                all_output = ""
                async for chunk in collector.stream():
                    all_output += chunk
                
                yield {
                    "type": "complete",
                    "content": all_output,
                    "execution_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"处理消息失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"处理失败: {str(e)}"
            }
    
    async def _execute_with_streaming(
        self, 
        enhanced_query: str, 
        message: str, 
        session_id: str,
        collector: StreamingOutputCollector
    ):
        """执行三步流程并流式输出"""
        try:
            # 第一步：决策分析
            logger.info("执行第一步：决策分析")
            task_plan = await self.graph.decision_agent.analyze_request(enhanced_query)

            # 检查是否是简单任务
            if task_plan.get("complexity") == "simple" and task_plan.get("direct_answer"):
                logger.info("简单任务，直接回答")
                
                # 直接输出答案作为最终答案
                await collector.add(UnifiedOutput(
                    step=3,  # 直接标记为最终答案
                    step_name="快速响应",
                    content=task_plan["direct_answer"],
                    metadata={"complexity": "simple"}
                ))
                
                # 保存到会话
                session_manager.add_message(
                    session_id, 
                    "assistant", 
                    task_plan["direct_answer"],
                    {"complexity": "simple", "direct_response": True}
                )
                
                return
            
            # 格式化第一步输出，包含推理过程
            step1_content = self._format_task_plan_with_reasoning(task_plan)
            
            # 立即输出第一步结果
            await collector.add(UnifiedOutput(
                step=1,
                step_name="决策分析推理",
                content=step1_content,
                metadata={"task_plan": task_plan}
            ))
            
            # 第二步：创建专家执行
            logger.info("执行第二步：专家执行")
            
            # 根据复杂度选择创建专家的方式
            complexity = task_plan.get("complexity", "medium")
            
            if complexity == "hard":
                # Hard任务：使用协调Agent创建8-12个专家
                experts = await self._create_experts_by_complexity(task_plan, message)
            else:
                # Easy/Medium任务：优先使用原有的搜索对比专家模式
                experts = await self._create_search_comparison_experts(task_plan, message)
            
            # 并行执行专家
            expert_tasks = []
            for expert in experts:
                task = self._execute_expert_with_output(
                    expert, message, enhanced_query, collector
                )
                expert_tasks.append(task)
            
            # 等待所有专家完成
            expert_results = await asyncio.gather(*expert_tasks, return_exceptions=True)
            
            # 处理结果
            valid_results = []
            for i, result in enumerate(expert_results):
                if isinstance(result, Exception):
                    logger.error(f"专家执行失败: {result}")
                    error_result = {
                        "agent_id": f"expert_{i}",
                        "specialty": "执行失败",
                        "status": "error",
                        "summary": f"错误: {str(result)}",
                        "result": "",
                        "tools_used": []
                    }
                    valid_results.append(error_result)
                elif result:
                    valid_results.append(result)
            
            # 第三步：整合最终答案
            logger.info("执行第三步：最终整合")
            final_answer = await self._integrate_with_comparison(valid_results)
            
            # 输出最终答案
            await collector.add(UnifiedOutput(
                step=3,
                step_name="最终答案",
                content=final_answer,
                metadata={"expert_count": len(valid_results)}
            ))
            
            # 保存到会话
            session_manager.add_message(session_id, "assistant", final_answer, {
                "task_plan": task_plan,
                "expert_results": valid_results
            })
            
        except Exception as e:
            logger.error(f"执行流程失败: {e}", exc_info=True)
            await collector.add(UnifiedOutput(
                step=-1,
                step_name="错误",
                content=f"执行失败: {str(e)}"
            ))
    
    def _format_task_plan_with_reasoning(self, task_plan: Dict) -> str:
        """格式化任务计划，包含推理过程"""
        complexity = task_plan.get("complexity", "medium")
        lines = [
            f"任务复杂度: {complexity}",
            f"任务拆解: {len(task_plan.get('sub_tasks', []))} 个子任务"
        ]
        
        # 添加推理过程（非simple任务）
        reasoning = task_plan.get("reasoning_process")
        if reasoning and complexity != "simple":
            lines.append(f"\n【决策推理过程】")
            lines.append(reasoning)
        
        # 列出子任务
        lines.append("\n【子任务列表】")
        for i, task in enumerate(task_plan.get('sub_tasks', []), 1):
            lines.append(f"  {i}. {task.get('task_type')}: {task.get('description')}")
            if task.get('required_tools'):
                lines.append(f"     需要工具: {', '.join(task['required_tools'])}")
        
        # 添加执行策略
        if task_plan.get("execution_mode"):
            lines.append(f"\n执行模式: {task_plan['execution_mode']}")
        
        if task_plan.get("risk_assessment"):
            lines.append(f"风险评估: {task_plan['risk_assessment']}")
        
        return "\n".join(lines)
    
    async def _create_experts_by_complexity(self, task_plan: Dict, message: str) -> List:
        """根据复杂度创建专家团队（用于Hard任务）"""
        complexity = task_plan.get("complexity", "medium")
        
        # 直接调用协调Agent的创建方法
        experts = await self.graph.coordination_agent.create_expert_agents(task_plan)
        
        # 设置原始查询
        for expert in experts:
            expert.original_query = message
        
        logger.info(f"为{complexity}任务创建了 {len(experts)} 个专家")
        return experts
    
    async def _create_search_comparison_experts(self, task_plan: Dict, message: str) -> List:
        """创建搜索对比专家（保留原有功能）"""
        from .agents.expert_agent import MCPMandatoryExpertAgent
        
        experts = []
        all_tools = self.graph.coordination_agent.available_tools
        
        if not all_tools:
            logger.warning("没有可用的MCP工具，创建纯LLM专家")
            config = {
                "agent_id": "llm_expert",
                "specialty": "LLM助手",
                "task_description": "使用语言模型回答问题",
                "all_tools": [],
                "mandatory_tools": [],
                "original_query": message,
                "complexity": task_plan.get("complexity", "medium")
            }
            experts.append(MCPMandatoryExpertAgent(self.graph.llm_manager, config))
            return experts
        
        # 检查是否需要搜索
        needs_search = any(
            keyword in str(task_plan).lower() 
            for keyword in ["搜索", "查找", "查询", "search", "find"]
        )
        
        if needs_search:
            logger.info("创建双搜索引擎对比专家")
            
            # Tavily搜索专家
            tavily_tools = [t for t in all_tools if 'tavily' in t.name.lower()]
            if tavily_tools:
                tavily_config = {
                    "agent_id": "tavily_search_expert",
                    "specialty": "Tavily搜索引擎",
                    "task_description": f"使用Tavily搜索: {message}",
                    "all_tools": all_tools,
                    "mandatory_tools": ["tavily-search"],
                    "original_query": message,
                    "complexity": task_plan.get("complexity", "medium")
                }
                experts.append(MCPMandatoryExpertAgent(self.graph.llm_manager, tavily_config))
            
            # Bing搜索专家
            bing_tools = [t for t in all_tools if 'bing' in t.name.lower()]
            if bing_tools:
                bing_config = {
                    "agent_id": "bing_search_expert",
                    "specialty": "Bing搜索引擎",
                    "task_description": f"使用Bing搜索: {message}",
                    "all_tools": all_tools,
                    "mandatory_tools": ["bing_search"],
                    "original_query": message,
                    "complexity": task_plan.get("complexity", "medium")
                }
                experts.append(MCPMandatoryExpertAgent(self.graph.llm_manager, bing_config))
        
        # 添加其他任务专家
        for i, subtask in enumerate(task_plan.get("sub_tasks", [])):
            if "搜索" not in subtask.get("task_type", ""):
                mandatory_tools = self._determine_mandatory_tools(subtask)
                
                if any('context7' in t.lower() or 'library' in t.lower() or 'docs' in t.lower() 
                       for t in mandatory_tools):
                    if 'resolve-library-id' not in mandatory_tools:
                        mandatory_tools.append('resolve-library-id')
                    if 'get-library-docs' not in mandatory_tools:
                        mandatory_tools.append('get-library-docs')
                
                config = {
                    "agent_id": f"expert_{i}_{subtask.get('task_type', 'general')}",
                    "specialty": "文档查询专家" if any('context7' in t or 'library' in t for t in mandatory_tools) else subtask.get("task_type", "通用专家"),
                    "task_description": subtask.get("description", ""),
                    "all_tools": all_tools,
                    "mandatory_tools": mandatory_tools,
                    "original_query": message,
                    "complexity": task_plan.get("complexity", "medium")
                }
                experts.append(MCPMandatoryExpertAgent(self.graph.llm_manager, config))
        
        # 如果没有专家，创建通用专家
        if not experts:
            # 选择可用的搜索工具
            search_tools = [t.name for t in all_tools if 'search' in t.name.lower() or 'tavily' in t.name.lower() or 'bing' in t.name.lower()][:2]
            
            config = {
                "agent_id": "general_expert",
                "specialty": "通用助手",
                "task_description": "回答用户问题",
                "all_tools": all_tools,
                "mandatory_tools": search_tools if search_tools else [],
                "original_query": message,
                "complexity": task_plan.get("complexity", "medium")
            }
            experts.append(MCPMandatoryExpertAgent(self.graph.llm_manager, config))
        
        logger.info(f"创建了 {len(experts)} 个专家Agent")
        return experts
    
    def _determine_mandatory_tools(self, subtask: Dict) -> List[str]:
        """确定必须使用的MCP工具"""
        task_type = subtask.get("task_type", "").lower()
        description = subtask.get("description", "").lower()
        
        tools = []
        
        if any(word in description + task_type for word in ["搜索", "查找", "查询", "search", "find"]):
            tools.append("search")
        
        if any(word in description + task_type for word in ["网页", "链接", "url", "网站", "page"]):
            tools.append("fetch")
        
        if any(word in description + task_type for word in ["文档", "库", "代码", "library", "docs", "context7"]):
            tools.extend(["resolve-library-id", "get-library-docs"])
        
        if any(word in description + task_type for word in ["论文", "学术", "研究", "paper", "research"]):
            tools.append("arxiv-mcp-server")
        
        if any(word in description + task_type for word in ["时间", "日期", "时区", "time", "date"]):
            tools.append("time")
        
        if not tools:
            tools.append("search")
        
        return tools
    
    async def _execute_expert_with_output(
        self, 
        expert, 
        message: str, 
        enhanced_query: str,
        collector: StreamingOutputCollector
    ) -> Dict:
        """执行专家并输出结果"""
        try:
            result = await expert.execute_with_mandatory_mcp(message, enhanced_query)
            
            # 立即输出专家结果
            await collector.add(UnifiedOutput(
                step=2,
                step_name=f"专家执行 - {result.get('agent_id', 'unknown')}",
                content=result.get('summary', '无结果'),
                metadata={
                    "mcp_tools": result.get('tools_used', []),
                    "specialty": result.get('specialty', ''),
                    "full_result": result,
                    "model_used": result.get('model_used', {})
                }
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"专家执行失败: {e}")
            error_result = {
                "agent_id": "error",
                "specialty": "执行失败",
                "status": "error",
                "summary": f"错误: {str(e)}",
                "result": "",
                "tools_used": []
            }
            
            await collector.add(UnifiedOutput(
                step=2,
                step_name="专家执行 - 错误",
                content=f"执行失败: {str(e)}",
                metadata={}
            ))
            
            return error_result
    
    def _needs_time_info(self, message: str) -> bool:
        """判断是否需要时间信息"""
        time_keywords = [
            "最新", "最近", "今天", "昨天", "今日", "现在", "当前",
            "本周", "本月", "今年", "实时", "更新", "新闻",
            "latest", "recent", "today", "current", "now"
        ]
        return any(keyword in message.lower() for keyword in time_keywords)
    
    async def _get_time_from_mcp(self) -> str:
        """通过time-MCP获取准确时间"""
        try:
            time_tools = [
                tool for tool in self.graph.coordination_agent.available_tools
                if 'time' in tool.name.lower() or 'get_current_time' in tool.name.lower()
            ]
            
            if time_tools:
                logger.info("调用time-MCP获取准确时间")
                tool = time_tools[0]
                # 不传递timezone参数，让工具使用默认值
                try:
                    result = await tool.ainvoke({})
                except Exception as e:
                    logger.debug(f"无参数调用失败: {e}")
                    # 尝试其他可能的参数
                    try:
                        result = await tool.ainvoke({"input": ""})
                    except:
                        result = None
                
                if result:
                    return f"[MCP时间] {result}"
            
            return f"[系统时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        except Exception as e:
            logger.error(f"time-MCP调用失败: {e}")
            return f"[系统时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _build_enhanced_query(self, message: str, context: str, time_context: str) -> str:
        """构建增强查询"""
        parts = []
        
        if time_context:
            parts.append(time_context)
        
        if context:
            parts.append(f"对话历史:\n{context}")
        
        parts.append(f"当前问题: {message}")
        
        return "\n\n".join(parts)
    
    async def _integrate_with_comparison(self, expert_results: List[Dict]) -> str:
        """整合结果并对比不同搜索引擎"""
        search_results = {}
        other_results = []
        
        # 分类结果
        for result in expert_results:
            if "search" in result.get("agent_id", "").lower():
                engine = result.get("specialty", "未知引擎")
                search_results[engine] = result
            else:
                other_results.append(result)
        
        # 构建对比
        comparison = []
        if len(search_results) > 1:
            comparison.append("【搜索引擎对比分析】")
            for engine, result in search_results.items():
                tools = ", ".join(result.get("tools_used", []))
                model_info = result.get("model_used", {})
                model_name = model_info.get("model_name", "unknown")
                
                comparison.append(f"\n{engine} (工具: {tools}, 模型: {model_name}):")
                comparison.append(result.get("summary", "无结果"))
        
        # 调用决策Agent进行最终整合
        final_integration = await self.graph.decision_agent.integrate_results(expert_results)
        
        if comparison:
            return "\n".join(comparison) + "\n\n" + final_integration
        
        return final_integration
    
    async def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self.initialized,
            "available_mcp_servers": self.graph.coordination_agent.mcp_manager.list_available_servers(),
            "available_tools": len(self.graph.coordination_agent.available_tools) if hasattr(self.graph.coordination_agent, 'available_tools') else 0,
            "llm_models": self.graph.llm_manager.list_available_models(),
            "active_sessions": len(session_manager.sessions)
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            await self.graph.coordination_agent.cleanup()
            logger.info("Agent系统资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

def create_agent() -> LangGraphMCPAgent:
    """创建Agent实例"""
    return LangGraphMCPAgent()