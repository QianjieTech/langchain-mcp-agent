#src\langgraph_mcp_agent\utils\logger.py
"""日志配置工具"""
import sys
import logging
from pathlib import Path
from datetime import datetime
import structlog
from typing import Any, Dict, List, Optional

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
):
    """配置结构化日志"""
    
    # 创建日志目录
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置处理器
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # 添加输出处理器
    if console_output:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # 配置structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置标准logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # 如果需要写入文件
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)
    
    return structlog.get_logger()

class AgentLogger:
    """Agent专用日志器"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = structlog.get_logger().bind(agent_id=agent_id)
        self.execution_logs = []
    
    def log_execution_start(self, task: str):
        """记录执行开始"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "execution_start",
            "task": task
        }
        self.execution_logs.append(log_entry)
        self.logger.info("执行开始", task=task)
    
    def log_tool_call(self, tool_name: str, parameters: dict, result: Any):
        """记录工具调用"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "tool_call",
            "tool": tool_name,
            "parameters": parameters,
            "result": str(result)[:200]  # 限制长度
        }
        self.execution_logs.append(log_entry)
        self.logger.debug("工具调用", tool=tool_name)
    
    def log_execution_end(self, status: str, result: Optional[str] = None, error: Optional[str] = None):
        """记录执行结束"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "execution_end",
            "status": status,
            "result": result[:500] if result else None,
            "error": error
        }
        self.execution_logs.append(log_entry)
        
        if status == "success":
            self.logger.info("执行成功")
        else:
            self.logger.error("执行失败", error=error)
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """获取执行跟踪"""
        return self.execution_logs