#src\langgraph_mcp_agent\utils\exceptions.py
"""异常类"""

class AgentException(Exception):
    """Agent基础异常"""
    pass

class LLMException(AgentException):
    """LLM调用异常"""
    pass

class MCPException(AgentException):
    """MCP相关异常"""
    pass

class ToolExecutionException(AgentException):
    """工具执行异常"""
    pass

class TaskPlanningException(AgentException):
    """任务规划异常"""
    pass

class CoordinationException(AgentException):
    """协调异常"""
    pass

class IntegrationException(AgentException):
    """结果整合异常"""
    pass

class ConfigurationException(AgentException):
    """配置异常"""
    pass

class TimeoutException(AgentException):
    """超时异常"""
    pass

class ResourceException(AgentException):
    """资源异常"""
    pass

def handle_agent_exception(func):
    """Agent异常处理装饰器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AgentException:
            raise
        except Exception as e:
            raise AgentException(f"Agent执行失败: {str(e)}") from e
    return wrapper