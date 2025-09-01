#src\langgraph_mcp_agent\graph\state_graph.py
"""LangGraph状态图定义"""
from typing import TypedDict, List, Dict, Any, Annotated
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from ..agents.decision_agent import DecisionAgent
from ..agents.coordination_agent import CoordinationAgent
from ..core.llm_manager import LLMConfigManager
from ..core.data_processor import DataProcessor
from ..utils.logger import AgentLogger
import structlog

logger = structlog.get_logger()

class AgentState(TypedDict):
    """Agent系统状态定义"""
    user_query: str
    task_plan: Dict[str, Any]
    expert_agents: List[Any]
    expert_results: List[Dict[str, Any]]
    final_answer: str
    messages: Annotated[List[BaseMessage], add_messages]
    errors: List[str]
    current_step: str
    mcp_tools_initialized: bool
    execution_start_time: str
    execution_end_time: str
    execution_trace: List[Dict[str, Any]]

class MultiAgentGraph:
    """多Agent系统状态"""
    
    def __init__(self):
        self.llm_manager = LLMConfigManager()
        self.decision_agent = DecisionAgent(self.llm_manager)
        self.coordination_agent = CoordinationAgent(self.llm_manager)
        self.data_processor = DataProcessor()
        self.graph = self._build_graph()
        self.system_logger = AgentLogger("system")
        
    def _build_graph(self) -> StateGraph:
        """构建状态图"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("initialize_mcp", self.initialize_mcp_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("coordinate", self.coordinate_node)
        workflow.add_node("execute", self.execute_node)
        workflow.add_node("integrate", self.integrate_node)
        
        # 设置入口
        workflow.set_entry_point("initialize_mcp")
        
        # 添加边
        workflow.add_edge("initialize_mcp", "analyze")
        workflow.add_edge("analyze", "coordinate")
        workflow.add_edge("coordinate", "execute")
        workflow.add_edge("execute", "integrate")
        workflow.add_edge("integrate", END)
        
        return workflow.compile()
    
    async def initialize_mcp_node(self, state: AgentState) -> AgentState:
        """初始化MCP工具节点"""
        logger.info("初始化MCP工具")
        state["execution_start_time"] = datetime.now().isoformat()
        
        try:
            await self.coordination_agent.initialize_tools()
            state["mcp_tools_initialized"] = True
            state["current_step"] = "mcp_initialized"
            
            # 记录执行跟踪
            state["execution_trace"].append({
                "step": "initialize_mcp",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "tools_count": len(self.coordination_agent.available_tools)
            })
            
            logger.info("MCP工具初始化成功")
        except Exception as e:
            logger.error(f"MCP工具初始化失败: {e}")
            state["errors"].append(f"MCP初始化错误: {str(e)}")
            state["mcp_tools_initialized"] = False
            
            state["execution_trace"].append({
                "step": "initialize_mcp",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        return state
    
    async def analyze_node(self, state: AgentState) -> AgentState:
        """决策分析节点"""
        logger.info("执行决策分析", query=state["user_query"])
        
        try:
            task_plan = await self.decision_agent.analyze_request(state["user_query"])
            state["task_plan"] = task_plan
            state["current_step"] = "analyze_complete"
            
            state["execution_trace"].append({
                "step": "analyze",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_plan": task_plan
            })
            
        except Exception as e:
            logger.error(f"决策分析失败: {e}")
            state["errors"].append(f"决策分析错误: {str(e)}")
            
            state["execution_trace"].append({
                "step": "analyze",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        return state
    
    async def coordinate_node(self, state: AgentState) -> AgentState:
        """协调创建节点"""
        logger.info("创建专家Agent团队")
        
        try:
            expert_agents = await self.coordination_agent.create_expert_agents(
                state["task_plan"]
            )
            state["expert_agents"] = expert_agents
            state["current_step"] = "coordinate_complete"
            
            state["execution_trace"].append({
                "step": "coordinate",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "experts_created": len(expert_agents)
            })
            
        except Exception as e:
            logger.error(f"协调创建失败: {e}")
            state["errors"].append(f"协调错误: {str(e)}")
            
            state["execution_trace"].append({
                "step": "coordinate",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        return state
    
    async def execute_node(self, state: AgentState) -> AgentState:
        """执行任务节点"""
        logger.info("执行专家任务")
        
        try:
            execution_mode = state["task_plan"].get("execution_mode", "concurrent")
            results = await self.coordination_agent.execute_tasks(execution_mode)
            state["expert_results"] = results
            state["current_step"] = "execute_complete"
            
            # 合并结果
            merged_results = self.data_processor.merge_expert_results(results)
            
            state["execution_trace"].append({
                "step": "execute",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "execution_mode": execution_mode,
                "results_summary": merged_results
            })
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            state["errors"].append(f"执行错误: {str(e)}")
            
            state["execution_trace"].append({
                "step": "execute",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        return state
    
    async def integrate_node(self, state: AgentState) -> AgentState:
        """结果整合节点"""
        logger.info("整合执行结果")
        
        try:
            final_answer = await self.decision_agent.integrate_results(
                state["expert_results"]
            )
            state["final_answer"] = final_answer
            state["current_step"] = "complete"
            state["execution_end_time"] = datetime.now().isoformat()
            
            state["execution_trace"].append({
                "step": "integrate",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "answer_length": len(final_answer)
            })
            
        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            state["errors"].append(f"整合错误: {str(e)}")
            state["final_answer"] = "抱歉，处理您的请求时出现了错误。"
            state["execution_end_time"] = datetime.now().isoformat()
            
            state["execution_trace"].append({
                "step": "integrate",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            
        finally:
            # 清理MCP资源
            await self.coordination_agent.cleanup()
        
        return state
    
    async def run(self, user_query: str) -> str:
        """运行Agent系统（简单接口）"""
        result = await self.run_with_details(user_query)
        return result.get("final_answer", "处理失败")
    
    async def run_with_details(self, user_query: str) -> Dict[str, Any]:
        """运行Agent系统（返回详细信息）"""
        initial_state = AgentState(
            user_query=user_query,
            task_plan={},
            expert_agents=[],
            expert_results=[],
            final_answer="",
            messages=[],
            errors=[],
            current_step="start",
            mcp_tools_initialized=False,
            execution_start_time="",
            execution_end_time="",
            execution_trace=[]
        )
        
        logger.info("启动Agent系统", query=user_query)
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state["errors"]:
                logger.warning("执行过程中出现错误", errors=final_state["errors"])
            
            # 计算执行时间
            if final_state["execution_start_time"] and final_state["execution_end_time"]:
                start = datetime.fromisoformat(final_state["execution_start_time"])
                end = datetime.fromisoformat(final_state["execution_end_time"])
                execution_time = (end - start).total_seconds()
            else:
                execution_time = None
            
            return {
                "final_answer": final_state["final_answer"],
                "task_plan": final_state["task_plan"],
                "expert_results": final_state["expert_results"],
                "errors": final_state["errors"],
                "execution_trace": final_state["execution_trace"],
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"系统运行失败: {e}")
            return {
                "final_answer": f"系统处理失败: {str(e)}",
                "task_plan": {},
                "expert_results": [],
                "errors": [str(e)],
                "execution_trace": [],
                "execution_time": None
            }