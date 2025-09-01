#src\langgraph_mcp_agent\core\session_manager.py
"""会话管理器"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import structlog

logger = structlog.get_logger()

class SessionManager:
    """会话管理器，管理多轮对话上下文"""
    
    def __init__(self, max_sessions: int = 1000, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
    def create_session(self, session_id: Optional[str] = None) -> str:
        """创建新会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # 清理过期会话
        self._cleanup_expired_sessions()
        
        # 如果达到最大会话数，删除最老的会话
        if len(self.sessions) >= self.max_sessions:
            oldest = min(self.sessions.items(), key=lambda x: x[1]['created_at'])
            del self.sessions[oldest[0]]
        
        self.sessions[session_id] = {
            'id': session_id,
            'messages': [],
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'metadata': {}
        }
        
        logger.info(f"创建新会话: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session['last_accessed'] = datetime.now()
            return session
        return None
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """添加消息到会话"""
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session(session_id)
            session = self.sessions[session_id]
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        session['messages'].append(message)
        
        # 限制消息历史长度
        max_messages = 50
        if len(session['messages']) > max_messages:
            session['messages'] = session['messages'][-max_messages:]
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取会话消息历史"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session['messages']
        if limit:
            return messages[-limit:]
        return messages
    
    def get_context_for_agent(self, session_id: str, max_context_messages: int = 10) -> str:
        """获取供Agent使用的上下文"""
        messages = self.get_messages(session_id, limit=max_context_messages)
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages:
            role_label = "用户" if msg['role'] == 'user' else "助手"
            context_parts.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.now()
        expired_ids = []
        
        for session_id, session in self.sessions.items():
            if now - session['last_accessed'] > self.session_timeout:
                expired_ids.append(session_id)
        
        for session_id in expired_ids:
            del self.sessions[session_id]
            logger.debug(f"清理过期会话: {session_id}")
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False

# 全局会话管理器实例
session_manager = SessionManager()