#src\langgraph_mcp_agent\core\mcp_manager.py
"""MCP集成管理器"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
import structlog

logger = structlog.get_logger()

class NativeMCPManager:
    """原生MCP管理器，使用langchain-mcp-adapters"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "mcp_config.json"
        
        self.config = self._load_config(config_path)
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载MCP配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            config = {"mcp_servers": {}}
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            config = {"mcp_servers": {}}
        
        # 处理配置，确保格式正确
        processed_servers = {}
        for server_name, server_config in config.get("mcp_servers", {}).items():
            clean_config = {
                "command": server_config.get("command"),
                "args": server_config.get("args", []),
                "transport": server_config.get("transport", "stdio")
            }
            
            # 处理URL类型的配置（用于HTTP传输）
            if "url" in server_config:
                clean_config["url"] = server_config["url"]
                clean_config["transport"] = "streamable_http"
            
            # 处理环境变量
            if "env" in server_config:
                env_dict = {}
                for key, value in server_config["env"].items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        env_value = os.getenv(env_var, "")
                        if env_value:
                            env_dict[key] = env_value
                        else:
                            logger.warning(f"环境变量 {env_var} 未设置")
                    else:
                        env_dict[key] = value
                
                if env_dict:
                    clean_config["env"] = env_dict
            
            processed_servers[server_name] = clean_config
            logger.debug(f"配置服务器 {server_name}: {clean_config}")
        
        return {"mcp_servers": processed_servers}
    
    async def initialize(self) -> List[BaseTool]:
        """初始化MCP客户端并获取所有工具"""
        try:
            if not self.config["mcp_servers"]:
                logger.warning("没有配置MCP服务器")
                return []
            
            # 创建MultiServerMCPClient
            self.mcp_client = MultiServerMCPClient(self.config["mcp_servers"])
            
            # 获取所有工具
            self.tools = await self.mcp_client.get_tools()
            
            logger.info(f"成功加载 {len(self.tools)} 个MCP工具",
                       servers=list(self.config["mcp_servers"].keys()))
            
            # 打印工具详情用于调试
            for tool in self.tools:
                logger.debug(f"工具: {tool.name} - {tool.description[:100] if tool.description else 'No description'}")
            
            return self.tools
            
        except Exception as e:
            logger.error(f"初始化MCP客户端失败: {e}", exc_info=True)
            # 返回空列表而不是抛出异常，允许系统继续运行
            self.tools = []
            return []
    
    async def get_tools_for_server(self, server_name: str) -> List[BaseTool]:
        """获取特定服务器的工具"""
        if not self.mcp_client:
            await self.initialize()
        
        try:
            tools = []
            # 根据服务器名称筛选工具
            if server_name in self.config["mcp_servers"]:
                # 这里需要根据实际的工具命名规则来判断
                # 通常工具名称会包含服务器名称的标识
                for tool in self.tools:
                    # 简单的启发式方法：如果工具名包含服务器名
                    if server_name.lower() in tool.name.lower():
                        tools.append(tool)
                    # 或者根据工具描述
                    elif tool.description and server_name.lower() in tool.description.lower():
                        tools.append(tool)
                
            return tools
        except Exception as e:
            logger.error(f"获取服务器 {server_name} 的工具失败: {e}")
            return []
    
    async def get_tools_by_type(self, tool_types: List[str]) -> List[BaseTool]:
        """根据类型获取工具"""
        if not self.tools:
            await self.initialize()
        
        filtered_tools = []
        added_tools = set()  # 避免重复添加
        
        for tool_type in tool_types:
            tool_type_lower = tool_type.lower()
            for tool in self.tools:
                if tool.name not in added_tools:
                    # 检查工具名称
                    if tool_type_lower in tool.name.lower():
                        filtered_tools.append(tool)
                        added_tools.add(tool.name)
                    # 检查工具描述
                    elif tool.description and tool_type_lower in tool.description.lower():
                        filtered_tools.append(tool)
                        added_tools.add(tool.name)
        
        return filtered_tools
    
    def list_available_servers(self) -> List[str]:
        """列出所有可用的MCP服务器"""
        return list(self.config["mcp_servers"].keys())
    
    def list_available_tools(self) -> List[Dict[str, str]]:
        """列出所有可用工具的信息"""
        return [
            {
                "name": tool.name,
                "description": tool.description if tool.description else "No description available"
            }
            for tool in self.tools
        ]
    
    async def cleanup(self):
        """清理资源"""
        if self.mcp_client:
            try:
                # MultiServerMCPClient的清理逻辑
                self.mcp_client = None
                self.tools = []
                logger.info("MCP客户端资源已清理")
            except Exception as e:
                logger.error(f"清理MCP客户端失败: {e}")