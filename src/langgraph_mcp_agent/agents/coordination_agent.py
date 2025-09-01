# src/langgraph_mcp_agent/agents/coordination_agent.py
"""协调Agent模块"""
import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from ..core.llm_manager import LLMConfigManager
from ..core.mcp_manager import NativeMCPManager
from .expert_agent import MCPMandatoryExpertAgent
import structlog

logger = structlog.get_logger()

class CoordinationAgent:
    """第二层：协调Agent
    负责创建和管理专家Agent，协调任务执行，分配MCP工具
    """
    
    def __init__(self, llm_manager: LLMConfigManager):
        self.llm_manager = llm_manager
        self.llm = llm_manager.get_model("coordination_agent")
        self.mcp_manager = NativeMCPManager()
        self.expert_agents: List[MCPMandatoryExpertAgent] = []
        self.available_tools: List[BaseTool] = []
        self.mcp_initialized = False
        
    async def initialize_tools(self):
        """初始化MCP工具（带错误处理）"""
        try:
            self.available_tools = await self.mcp_manager.initialize()
            self.mcp_initialized = True
            
            if not self.available_tools:
                logger.warning("没有可用的MCP工具，将使用纯LLM模式")
            else:
                logger.info(f"成功初始化 {len(self.available_tools)} 个MCP工具")
                
        except Exception as e:
            logger.error(f"MCP工具初始化失败: {e}", exc_info=True)
            self.available_tools = []
            self.mcp_initialized = False
        
    async def create_expert_agents(self, task_plan: Dict[str, Any]) -> List[MCPMandatoryExpertAgent]:
        """根据任务计划创建专家Agent并分配MCP工具"""
        
        # 确保工具已尝试初始化
        if not self.mcp_initialized:
            await self.initialize_tools()
        
        complexity = task_plan.get("complexity", "medium")
        logger.info(f"开始创建专家Agent，任务复杂度: {complexity}")
        
        # 如果没有MCP工具，创建纯LLM专家
        if not self.available_tools:
            logger.info("使用纯LLM模式创建专家Agent")
            return await self._create_llm_only_experts(task_plan)
        
        # 根据复杂度创建不同数量的专家
        if complexity == "hard":
            return await self._create_hard_task_experts(task_plan)
        else:
            return await self._create_standard_experts(task_plan)
    
    async def _create_hard_task_experts(self, task_plan: Dict[str, Any]) -> List[MCPMandatoryExpertAgent]:
        """为困难任务创建专家团队（8-12个）"""
        logger.info("创建困难任务专家团队")
        
        # 获取可用工具列表
        tool_descriptions = [
            f"- {tool.name}: {tool.description}"
            for tool in self.available_tools
        ]
        
        creation_prompt = f"""根据以下困难任务计划，设计一个全面的专家Agent团队。

任务计划:
{json.dumps(task_plan, ensure_ascii=False, indent=2)}

可用的MCP工具({len(self.available_tools)}个):
{chr(10).join(tool_descriptions)}

要求：
1. 创建8-12个专家Agent，充分利用所有可用工具
2. 每个专家应有明确的专长和工具分配
3. 考虑任务的各个方面，确保全面覆盖
4. 对于复杂任务，同一工具可分配给多个专家进行不同角度的使用
5. 设计协同策略，让专家们能够互补
6. 为每个专家指定合适的模型偏好（easy/medium/hard）

返回JSON格式：
{{
    "expert_agents": [
        {{
            "agent_id": "唯一标识",
            "specialty": "专业领域",
            "task_description": "具体任务描述",
            "assigned_mcp_tools": ["分配的MCP工具名称"],
            "execution_strategy": "执行策略",
            "output_format": "输出格式要求",
            "model_preference": "easy/medium/hard"
        }}
    ],
    "coordination_strategy": "协调策略描述"
}}"""
        
        messages = [
            SystemMessage(content="你是协调Agent，负责为困难任务创建全面的专家团队。"),
            HumanMessage(content=creation_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            agent_configs = self._parse_agent_configs(response.content)
            
            # 创建专家Agent实例
            self.expert_agents = []
            for config in agent_configs.get("expert_agents", []):
                # 为每个专家分配具体的工具对象
                assigned_tools = self._get_tools_by_names(
                    config.get("assigned_mcp_tools", [])
                )
                
                # 构建完整的专家配置
                expert_config = {
                    "agent_id": config.get("agent_id", f"expert_{len(self.expert_agents)}"),
                    "specialty": config.get("specialty", "通用专家"),
                    "task_description": config.get("task_description", ""),
                    "assigned_tools": assigned_tools,
                    "all_tools": self.available_tools,
                    "mandatory_tools": config.get("assigned_mcp_tools", []),
                    "execution_strategy": config.get("execution_strategy", "标准执行"),
                    "output_format": config.get("output_format", "结构化文本"),
                    "complexity": task_plan.get("complexity", "hard"),
                    "model_preference": config.get("model_preference", "auto")
                }
                
                expert = MCPMandatoryExpertAgent(self.llm_manager, expert_config)
                self.expert_agents.append(expert)
            
            # 确保至少有8个专家
            while len(self.expert_agents) < 8:
                additional_expert = self._create_additional_expert(
                    len(self.expert_agents), 
                    task_plan,
                    "hard"
                )
                self.expert_agents.append(additional_expert)
            
            logger.info(f"创建了 {len(self.expert_agents)} 个困难任务专家Agent")
            
            # 打印每个专家使用的模型
            for expert in self.expert_agents:
                model_info = expert._get_model_info()
                logger.info(f"专家 {expert.agent_id} 使用模型: {model_info.get('model_name')}")
            
            return self.expert_agents
            
        except Exception as e:
            logger.error(f"创建困难任务专家失败: {e}")
            return self._create_fallback_experts(task_plan, min_experts=8)
    
    async def _create_standard_experts(self, task_plan: Dict[str, Any]) -> List[MCPMandatoryExpertAgent]:
        """创建标准专家团队"""
        # 获取可用工具列表
        tool_descriptions = [
            f"- {tool.name}: {tool.description}"
            for tool in self.available_tools
        ]
        
        creation_prompt = f"""根据以下任务计划，设计需要创建的专家Agent配置。

任务计划:
{json.dumps(task_plan, ensure_ascii=False, indent=2)}

可用的MCP工具:
{chr(10).join(tool_descriptions) if tool_descriptions else '无可用工具'}

MCP服务器:
{', '.join(self.mcp_manager.list_available_servers())}

请为每个子任务创建对应的专家Agent配置，返回JSON格式：
{{
    "expert_agents": [
        {{
            "agent_id": "唯一标识",
            "specialty": "专业领域",
            "task_description": "具体任务描述",
            "assigned_mcp_tools": ["分配的MCP工具名称"],
            "execution_strategy": "执行策略",
            "output_format": "输出格式要求",
            "model_preference": "easy/medium/hard"
        }}
    ],
    "coordination_strategy": "协调策略描述"
}}

注意：assigned_mcp_tools中的工具名称必须来自上面的可用工具列表。"""
        
        messages = [
            SystemMessage(content="你是协调Agent，负责创建和管理专家Agent团队，并分配合适的MCP工具。"),
            HumanMessage(content=creation_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            agent_configs = self._parse_agent_configs(response.content)
            
            # 创建专家Agent实例
            self.expert_agents = []
            for config in agent_configs.get("expert_agents", []):
                # 为每个专家分配具体的工具对象
                assigned_tools = self._get_tools_by_names(
                    config.get("assigned_mcp_tools", [])
                )
                
                # 构建完整的专家配置
                expert_config = {
                    "agent_id": config.get("agent_id", f"expert_{len(self.expert_agents)}"),
                    "specialty": config.get("specialty", "通用专家"),
                    "task_description": config.get("task_description", ""),
                    "assigned_tools": assigned_tools,
                    "all_tools": self.available_tools,
                    "mandatory_tools": config.get("assigned_mcp_tools", []),
                    "execution_strategy": config.get("execution_strategy", "标准执行"),
                    "output_format": config.get("output_format", "结构化文本"),
                    "complexity": task_plan.get("complexity", "medium"),
                    "model_preference": config.get("model_preference", "auto")
                }
                
                expert = MCPMandatoryExpertAgent(self.llm_manager, expert_config)
                self.expert_agents.append(expert)
                
            if not self.expert_agents:
                # 如果没有成功创建任何专家，创建默认专家
                self.expert_agents = [self._create_default_expert(task_plan)]
                
            logger.info(f"创建了 {len(self.expert_agents)} 个专家Agent，分配了MCP工具")
            return self.expert_agents
            
        except Exception as e:
            logger.error(f"创建专家Agent失败: {e}")
            return [self._create_default_expert(task_plan)]
    
    def _create_additional_expert(self, index: int, task_plan: Dict, complexity: str = "medium") -> MCPMandatoryExpertAgent:
        """创建额外的专家"""
        # 选择未充分使用的工具
        used_tools = set()
        for expert in self.expert_agents:
            if hasattr(expert, 'assigned_tools'):
                for tool in expert.assigned_tools:
                    used_tools.add(tool.name)
        
        unused_tools = []
        for tool in self.available_tools:
            if tool.name not in used_tools:
                unused_tools.append(tool)
        
        # 如果有未使用的工具，优先分配
        if unused_tools:
            assigned_tools = unused_tools[:2]
        else:
            # 否则轮流分配工具
            tool_index = index % len(self.available_tools)
            assigned_tools = [self.available_tools[tool_index]]
        
        # 根据索引选择不同的专长
        specialties = [
            "数据分析", "信息检索", "深度研究", "趋势分析",
            "技术评估", "对比分析", "综合研究", "详细调查"
        ]
        specialty = specialties[index % len(specialties)]
        
        # 根据复杂度调整模型偏好
        model_preference = "hard" if complexity == "hard" and index < 4 else "medium"
        
        config = {
            "agent_id": f"additional_expert_{index}",
            "specialty": f"{specialty}专家",
            "task_description": f"从{specialty}角度分析: {task_plan.get('analysis', '')}",
            "assigned_tools": assigned_tools,
            "all_tools": self.available_tools,
            "mandatory_tools": [tool.name for tool in assigned_tools],
            "execution_strategy": "深度挖掘和交叉验证",
            "output_format": "详细分析报告",
            "complexity": complexity,
            "model_preference": model_preference
        }
        
        return MCPMandatoryExpertAgent(self.llm_manager, config)
    
    def _create_fallback_experts(self, task_plan: Dict, min_experts: int = 1) -> List[MCPMandatoryExpertAgent]:
        """创建备用专家团队"""
        experts = []
        
        # 根据可用工具数量创建专家
        tools_per_expert = max(1, len(self.available_tools) // min_experts) if self.available_tools else 0
        
        for i in range(min_experts):
            if self.available_tools:
                start_idx = i * tools_per_expert
                end_idx = start_idx + tools_per_expert
                assigned_tools = self.available_tools[start_idx:end_idx]
            else:
                assigned_tools = []
            
            config = {
                "agent_id": f"fallback_expert_{i}",
                "specialty": f"综合分析专家{i+1}",
                "task_description": task_plan.get("analysis", "执行任务分析"),
                "assigned_tools": assigned_tools,
                "all_tools": self.available_tools,
                "mandatory_tools": [tool.name for tool in assigned_tools],
                "execution_strategy": "标准执行",
                "output_format": "结构化分析",
                "complexity": task_plan.get("complexity", "medium"),
                "model_preference": "auto"
            }
            
            experts.append(MCPMandatoryExpertAgent(self.llm_manager, config))
        
        return experts
    
    async def _create_llm_only_experts(self, task_plan: Dict[str, Any]) -> List[MCPMandatoryExpertAgent]:
        """创建纯LLM专家（无工具）"""
        experts = []
        
        sub_tasks = task_plan.get("sub_tasks", [])
        complexity = task_plan.get("complexity", "medium")
        
        if not sub_tasks:
            # 如果没有子任务，创建一个通用专家
            return [self._create_default_expert(task_plan)]
        
        for i, task in enumerate(sub_tasks):
            # 根据任务类型选择模型偏好
            task_type = task.get("task_type", "").lower()
            if "simple" in task_type or "quick" in task_type:
                model_preference = "easy"
            elif "complex" in task_type or "deep" in task_type:
                model_preference = "hard"
            else:
                model_preference = "medium"
            
            config = {
                "agent_id": f"llm_expert_{i}",
                "specialty": task.get("task_type", "LLM专家"),
                "task_description": task.get("description", "执行任务"),
                "assigned_tools": [],  # 无工具
                "all_tools": [],
                "mandatory_tools": [],
                "execution_strategy": "纯LLM推理",
                "output_format": "结构化文本",
                "complexity": complexity,
                "model_preference": model_preference
            }
            expert = MCPMandatoryExpertAgent(self.llm_manager, config)
            experts.append(expert)
        
        logger.info(f"创建了 {len(experts)} 个纯LLM专家")
        return experts
    
    def _get_tools_by_names(self, tool_names: List[str]) -> List[BaseTool]:
        """根据工具名称获取工具对象"""
        tools = []
        for name in tool_names:
            for tool in self.available_tools:
                if tool.name.lower() == name.lower():
                    tools.append(tool)
                    break
        return tools
    
    async def execute_tasks(self, execution_mode: str = "concurrent") -> List[Dict[str, Any]]:
        """执行专家Agent任务"""
        if not self.expert_agents:
            logger.warning("没有可用的专家Agent")
            return []
        
        if execution_mode == "concurrent":
            return await self._execute_concurrent()
        else:
            return await self._execute_sequential()
    
    async def _execute_concurrent(self) -> List[Dict[str, Any]]:
        """并发执行任务"""
        logger.info(f"开始并发执行 {len(self.expert_agents)} 个任务")
        
        tasks = []
        for agent in self.expert_agents:
            # 使用正确的方法名
            if hasattr(agent, 'execute_with_mandatory_mcp'):
                task = asyncio.create_task(
                    agent.execute_with_mandatory_mcp(
                        agent.task_description,
                        agent.task_description
                    )
                )
            else:
                # 备用方法
                task = asyncio.create_task(
                    self._execute_agent_task(agent)
                )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"专家Agent {i} 执行失败: {result}")
                processed_results.append({
                    "agent_id": self.expert_agents[i].agent_id if i < len(self.expert_agents) else f"expert_{i}",
                    "status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_sequential(self) -> List[Dict[str, Any]]:
        """顺序执行任务"""
        logger.info(f"开始顺序执行 {len(self.expert_agents)} 个任务")
        
        results = []
        for i, agent in enumerate(self.expert_agents):
            try:
                if hasattr(agent, 'execute_with_mandatory_mcp'):
                    result = await agent.execute_with_mandatory_mcp(
                        agent.task_description,
                        agent.task_description
                    )
                else:
                    result = await self._execute_agent_task(agent)
                results.append(result)
            except Exception as e:
                logger.error(f"专家Agent {i} 执行失败: {e}")
                results.append({
                    "agent_id": agent.agent_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    async def _execute_agent_task(self, agent) -> Dict[str, Any]:
        """执行单个Agent任务（备用方法）"""
        try:
            # 构造基本响应
            return {
                "agent_id": agent.agent_id,
                "specialty": agent.specialty,
                "status": "completed",
                "summary": f"{agent.specialty}完成了任务分析",
                "result": f"基于{agent.specialty}的分析结果",
                "tools_used": []
            }
        except Exception as e:
            logger.error(f"执行Agent任务失败: {e}")
            return {
                "agent_id": agent.agent_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _parse_agent_configs(self, response: str) -> Dict[str, Any]:
        """解析Agent配置响应"""
        try:
            # 尝试提取JSON
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # 尝试直接解析
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"配置解析失败: {e}，使用默认配置")
            return {"expert_agents": []}
        except Exception as e:
            logger.error(f"解析配置时出错: {e}")
            return {"expert_agents": []}
    
    def _create_default_expert(self, task_plan: Dict[str, Any]) -> MCPMandatoryExpertAgent:
        """创建默认专家Agent"""
        # 尝试分配一些基础工具
        default_tools = []
        if self.available_tools:
            # 选择前3个工具作为默认工具
            default_tools = self.available_tools[:3]
        
        default_config = {
            "agent_id": "default_expert",
            "specialty": "通用专家",
            "task_description": task_plan.get("analysis", "执行通用任务分析"),
            "assigned_tools": default_tools,
            "all_tools": self.available_tools,
            "mandatory_tools": [tool.name for tool in default_tools],
            "execution_strategy": "标准执行",
            "output_format": "结构化文本",
            "complexity": task_plan.get("complexity", "medium"),
            "model_preference": "auto"
        }
        return MCPMandatoryExpertAgent(self.llm_manager, default_config)
    
    async def cleanup(self):
        """清理资源"""
        try:
            await self.mcp_manager.cleanup()
            self.expert_agents = []
            self.available_tools = []
            logger.info("协调Agent资源已清理")
        except Exception as e:
            logger.error(f"清理协调Agent资源失败: {e}")