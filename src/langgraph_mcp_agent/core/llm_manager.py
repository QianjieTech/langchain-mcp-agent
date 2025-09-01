# src/langgraph_mcp_agent/core/llm_manager.py
"""LLM管理器模块"""
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from ..config.settings import settings, LLMConfig, ExpertModelConfig
import structlog

logger = structlog.get_logger()

class LLMConfigManager:
    """LLM配置管理器，统一管理所有模型调用"""
    
    def __init__(self):
        self._models_cache: Dict[str, BaseChatModel] = {}
        self._configs = settings.MODEL_MAPPINGS
        self._expert_models = settings.EXPERT_MODELS
        
    def get_model(self, agent_type: str) -> BaseChatModel:
        """获取指定类型的模型实例"""
        if agent_type in self._models_cache:
            return self._models_cache[agent_type]
        
        config = settings.get_llm_config(agent_type)
        model = self._create_model(config)
        self._models_cache[agent_type] = model
        
        logger.info(f"创建模型实例", agent_type=agent_type, model=config.model_name)
        return model
    
    def get_expert_model(self, task_type: str, complexity: str = "medium") -> BaseChatModel:
        """根据任务类型和复杂度获取专家模型"""
        model_id = self._select_expert_model(task_type, complexity)
        
        cache_key = f"expert_{model_id}"
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        config = settings.get_expert_model_config(model_id)
        model = self._create_model_from_expert_config(config)
        self._models_cache[cache_key] = model
        
        logger.info(f"创建专家模型", 
                   model_id=model_id, 
                   task_type=task_type,
                   complexity=complexity)
        return model
    
    # src/langgraph_mcp_agent/core/llm_manager.py - 修正 _select_expert_model 方法
    def _select_expert_model(self, task_type: str, complexity: str) -> str:
        """选择最适合的专家模型"""
        task_type_lower = task_type.lower()
        
        logger.debug(f"选择专家模型 - 任务类型: {task_type}, 复杂度: {complexity}")
        
        # 根据任务类型和复杂度选择模型
        selected_model = None
        
        # Easy任务：轻量快速模型
        if complexity == "easy":
            # 搜索类任务
            if any(keyword in task_type_lower for keyword in ["搜索", "查找", "search", "find", "查询"]):
                selected_model = "glm-4.5"  # 快速搜索
            # 代码类任务
            elif any(keyword in task_type_lower for keyword in ["代码", "code", "编程", "开发"]):
                selected_model = "qwen_30b"  # 轻量代码模型
            # 数据提取或快速分析
            else:
                selected_model = "glm-4.5-airx"  # 结构化输出
        
        # Medium任务：平衡性能
        elif complexity == "medium":
            # 需要推理的任务
            if any(keyword in task_type_lower for keyword in ["分析", "推理", "评估", "analysis", "reasoning"]):
                selected_model = "gemini-2.5-pro"  # 稳定推理
            # 综合类任务
            elif any(keyword in task_type_lower for keyword in ["综合", "对比", "比较", "comprehensive"]):
                selected_model = "gemini-2.5-pro"
            # 技术文档或专业内容
            elif any(keyword in task_type_lower for keyword in ["文档", "技术", "专业", "technical"]):
                selected_model = "glm-4.5"  # 精准输出
            else:
                selected_model = "gemini-2.5-pro"  # 默认使用稳定模型
        
        # Hard任务：最强模型
        elif complexity == "hard":
            # 深度研究或复杂推理
            if any(keyword in task_type_lower for keyword in ["深度", "研究", "deep", "research", "论文", "学术"]):
                selected_model = "claude-opus-4"  # 最强推理
            # 复杂代码或系统设计
            elif any(keyword in task_type_lower for keyword in ["架构", "系统", "设计", "architecture"]):
                selected_model = "claude-opus-4"
            # 数据分析专家
            elif any(keyword in task_type_lower for keyword in ["数据分析", "统计", "data analysis"]):
                selected_model = "gemini-2.5-pro"  # 数据处理能力强
            # 信息检索专家
            elif any(keyword in task_type_lower for keyword in ["信息检索", "information", "retrieval"]):
                selected_model = "glm-4.5"  # 快速检索
            else:
                selected_model = "gemini-2.5-pro"  # 默认高性能模型
        
        # 如果没有选择，使用默认
        if not selected_model:
            selected_model = settings.DEFAULT_EXPERT_MODEL
        
        logger.info(f"模型选择结果: {selected_model} (任务: {task_type}, 复杂度: {complexity})")
    
        return selected_model
    
    def _create_model(self, config: LLMConfig) -> BaseChatModel:
        """根据配置创建模型实例"""
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=180,
            max_retries=5
        )
    
    def _create_model_from_expert_config(self, config: ExpertModelConfig) -> BaseChatModel:
        """从专家配置创建模型"""
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=180,
            max_retries=5
        )
    
    def update_config(self, agent_type: str, **kwargs):
        """动态更新模型配置"""
        if agent_type in self._configs:
            self._configs[agent_type].update(kwargs)
            # 清除缓存，下次获取时重新创建
            if agent_type in self._models_cache:
                del self._models_cache[agent_type]
    
    def list_available_models(self) -> Dict[str, str]:
        """列出所有可用的模型"""
        base_models = {
            agent_type: config["model_name"] 
            for agent_type, config in self._configs.items()
        }
        
        expert_models = {
            f"expert_{model_id}": config["model_name"]
            for model_id, config in self._expert_models.items()
        }
        
        return {**base_models, **expert_models}
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        if model_id in self._expert_models:
            config = self._expert_models[model_id]
            return {
                "model_id": model_id,
                "model_name": config["model_name"],
                "description": config["description"],
                "capabilities": config.get("capabilities", []),
                "speed": config.get("speed", "medium"),
                "cost": config.get("cost", "medium"),
                "best_for": config.get("best_for", [])
            }
        return {}