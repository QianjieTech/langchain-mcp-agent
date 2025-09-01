# src/langgraph_mcp_agent/core/conversation_state.py
"""对话状态管理器"""
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import uuid

class ConversationStage(Enum):
    """对话阶段枚举"""
    INITIAL = "initial"  # 初始状态
    PLANNING = "planning"  # 计划制定中
    CONFIRMING = "confirming"  # 等待用户确认
    ADJUSTING = "adjusting"  # 调整计划中
    EXECUTING = "executing"  # 执行中
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消

class ConversationState:
    """对话状态"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_id = str(uuid.uuid4())
        self.stage = ConversationStage.INITIAL
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # 对话历史
        self.messages: List[Dict[str, Any]] = []
        
        # 任务计划
        self.current_plan: Optional[Dict[str, Any]] = None
        self.plan_history: List[Dict[str, Any]] = []  # 计划修改历史
        
        # 执行状态
        self.execution_started = False
        self.execution_results: Optional[List[Dict[str, Any]]] = None
        
        # 用户偏好
        self.user_preferences = {
            "preferred_complexity": None,
            "time_constraint": None,
            "detail_level": "medium"
        }
    
    def update_stage(self, new_stage: ConversationStage):
        """更新对话阶段"""
        self.stage = new_stage
        self.updated_at = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到历史"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
    
    def update_plan(self, new_plan: Dict[str, Any]):
        """更新当前计划"""
        if self.current_plan:
            self.plan_history.append({
                "plan": self.current_plan.copy(),
                "timestamp": datetime.now().isoformat()
            })
        self.current_plan = new_plan
        self.updated_at = datetime.now()
    
    def get_context_summary(self) -> str:
        """获取对话上下文摘要"""
        summary_parts = []
        
        # 当前阶段
        summary_parts.append(f"对话阶段: {self.stage.value}")
        
        # 计划概要
        if self.current_plan:
            complexity = self.current_plan.get("complexity", "unknown")
            task_count = len(self.current_plan.get("sub_tasks", []))
            summary_parts.append(f"当前计划: {complexity}复杂度, {task_count}个任务")
        
        # 修改历史
        if self.plan_history:
            summary_parts.append(f"计划已修改{len(self.plan_history)}次")
        
        return " | ".join(summary_parts)

class ConversationStateManager:
    """对话状态管理器"""
    def __init__(self):
        self.states: Dict[str, ConversationState] = {}
    
    def get_or_create_state(self, session_id: str) -> ConversationState:
        """获取或创建对话状态"""
        if session_id not in self.states:
            self.states[session_id] = ConversationState(session_id)
        return self.states[session_id]
    
    def get_state(self, session_id: str) -> Optional[ConversationState]:
        """获取对话状态"""
        return self.states.get(session_id)
    
    def clear_state(self, session_id: str):
        """清除对话状态"""
        if session_id in self.states:
            del self.states[session_id]
    
    def cleanup_old_states(self, hours: int = 24):
        """清理超时的对话状态"""
        from datetime import timedelta
        
        now = datetime.now()
        to_remove = []
        
        for session_id, state in self.states.items():
            if now - state.updated_at > timedelta(hours=hours):
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.states[session_id]

# 全局实例
conversation_state_manager = ConversationStateManager()