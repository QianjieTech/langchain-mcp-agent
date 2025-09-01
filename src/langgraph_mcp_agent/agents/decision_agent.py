# src/langgraph_mcp_agent/agents/decision_agent.py
"""决策分析推理Agent"""
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from ..core.llm_manager import LLMConfigManager
import structlog

logger = structlog.get_logger()

class DecisionAgent:
    """第一层：决策分析推理Agent
    负责分析用户请求，制定执行策略，整合结果
    """
    
    def __init__(self, llm_manager: LLMConfigManager):
        self.llm = llm_manager.get_model("decision_agent")
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """加载系统提示词"""
        return """你是系统的决策分析推理Agent，具备强大的推理能力。你的职责是：
1. 深入理解和分析用户查询
2. 评估任务复杂度和所需资源
3. 制定详细的任务分解计划
4. 指定每个子任务需要的专家类型和工具
5. 预测执行流程中的潜在问题
6. **为你的决策提供清晰的推理过程**
7. **为每个子任务建议合适的模型复杂度**
任务复杂度等级定义：
- simple: 简单问候、感谢等不需要查询的日常对话
- easy: 需要2-4个子任务，通常是单一维度的查询
- medium: 需要4-7个子任务，涉及多个方面的综合查询
- hard: 需要8-12个子任务，复杂的深度研究或多维度分析
模型选择建议：
- 简单搜索/查询任务: 使用快速轻量模型（glm-4.5-airx）
- 代码相关任务: 使用专门的代码模型（qwen_30b）
- 复杂推理任务: 使用强大模型（gpt5）
- 深度分析任务: 使用最强模型（claude-opus-4-1）


请以JSON格式返回执行计划：
{
    "analysis": "查询分析结果",
    "complexity": "simple/easy/medium/hard",
    "reasoning_process": "决策推理过程（非simple任务必填）",
    "direct_answer": "如果是simple复杂度，这里是直接答案，否则为null",
    "sub_tasks": [
        {
            "task_id": "唯一标识",
            "task_type": "任务类型",
            "description": "任务描述",
            "priority": "high/medium/low",
            "estimated_time": "预估时间(秒)",
            "required_tools": ["所需工具列表"],
            "dependencies": ["依赖的任务ID"],
            "success_criteria": "成功标准",
            "suggested_model_complexity": "easy/medium/hard"
        }
    ],
    "execution_mode": "concurrent/sequential",
    "risk_assessment": "风险评估",
    "expected_output": "预期输出描述"
}
决策推理要求：
- 对于easy任务：简要说明为什么这样分解任务
- 对于medium任务：详细解释任务分解逻辑和工具选择理由
- 对于hard任务：提供完整的推理链，包括任务依赖关系、执行顺序、风险评估等
记住：根据任务复杂度生成相应数量的子任务！"""
    
    async def analyze_request(self, user_query: str) -> Dict[str, Any]:
        """分析用户请求，制定执行计划"""
        try:
            # 快速检查是否是简单问候
            greetings = ["你好", "hi", "hello", "嗨", "您好", "早上好", "晚上好", "下午好", "谢谢", "感谢"]
            query_lower = user_query.lower()
            
            for greeting in greetings:
                if greeting in query_lower and len(user_query) < 20:
                    # 直接返回简单响应
                    return {
                        "analysis": "简单问候或感谢",
                        "complexity": "simple",
                        "reasoning_process": None,
                        "direct_answer": self._get_greeting_response(user_query),
                        "sub_tasks": [],
                        "execution_mode": "sequential",
                        "risk_assessment": "无风险",
                        "expected_output": "友好回应"
                    }
            
            # 复杂查询，使用LLM分析
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"用户查询: {user_query}\n\n请分析并制定执行计划。")
            ]
            
            response = await self.llm.ainvoke(messages)
            plan = self._parse_response(response.content)
            
            # 验证子任务数量
            plan = self._validate_task_count(plan)
            
            logger.info("决策分析完成", 
                       complexity=plan.get("complexity"),
                       task_count=len(plan.get("sub_tasks", [])))
            
            return plan
            
        except Exception as e:
            logger.error(f"决策分析失败: {e}")
            return self._create_fallback_plan(user_query)
    
    def _validate_task_count(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """验证并调整任务数量"""
        complexity = plan.get("complexity", "medium")
        task_count = len(plan.get("sub_tasks", []))
        
        # 定义任务数量范围
        task_ranges = {
            "simple": (0, 0),
            "easy": (2, 4),
            "medium": (4, 7),
            "hard": (8, 12)
        }
        
        min_tasks, max_tasks = task_ranges.get(complexity, (2, 7))
        
        # 如果任务数量不在范围内，调整复杂度或生成更多任务
        if complexity != "simple" and task_count < min_tasks:
            logger.warning(f"{complexity}任务数量不足({task_count}), 需要至少{min_tasks}个")
            # 这里可以触发重新生成或降低复杂度
            if task_count <= 2:
                plan["complexity"] = "easy"
            elif task_count <= 4:
                plan["complexity"] = "medium"
        
        return plan
    
    def _get_greeting_response(self, query: str) -> str:
        """生成简单问候响应"""
        query_lower = query.lower()
        if "谢" in query or "thank" in query_lower:
            return "不客气！有什么需要帮助的，随时告诉我。"
        elif "早上好" in query or "morning" in query_lower:
            return "早上好！今天有什么可以帮助您的吗？"
        elif "晚上好" in query or "evening" in query_lower:
            return "晚上好！有什么需要我协助的吗？"
        else:
            return "你好！很高兴为您服务。有什么可以帮助您的吗？"
    
    async def integrate_results(self, results: List[Dict[str, Any]]) -> str:
        """整合各专家Agent的执行结果"""
        integration_prompt = f"""请整合以下专家Agent的执行结果，生成完整、准确、用户友好的最终答案。

专家结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

整合要求：
1. 去除冗余和无效信息
2. 逻辑清晰，结构完整
3. 保证信息准确性
4. 优化用户体验
5. 如有必要，标注信息来源
6. 对比不同来源的信息，指出差异和一致性

请直接返回整合后的答案文本。"""
        
        messages = [
            SystemMessage(content="你是结果整合专家，负责将多个专家的输出整合成连贯的答案。"),
            HumanMessage(content=integration_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 提取JSON内容
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            # 返回简化的默认结构
            return {
                "analysis": response,
                "complexity": "medium",
                "reasoning_process": "解析失败，使用默认推理",
                "direct_answer": None,
                "sub_tasks": [],
                "execution_mode": "concurrent",
                "risk_assessment": "解析失败，使用默认配置"
            }
    
    def _create_fallback_plan(self, user_query: str) -> Dict[str, Any]:
        """创建备用执行计划"""
        return {
            "analysis": f"用户查询: {user_query}",
            "complexity": "medium",
            "reasoning_process": "使用备用计划，基于通用搜索策略",
            "direct_answer": None,
            "sub_tasks": [
                {
                    "task_id": "fallback_001",
                    "task_type": "general_search",
                    "description": "执行通用搜索和分析",
                    "priority": "high",
                    "estimated_time": "30",
                    "required_tools": ["web_search"],
                    "dependencies": [],
                    "success_criteria": "获取相关信息"
                }
            ],
            "execution_mode": "sequential",
            "risk_assessment": "使用备用计划，可能无法充分满足需求"
        }