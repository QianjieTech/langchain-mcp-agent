# src/langgraph_mcp_agent/config/settings.py
"""配置管理模块"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    """LLM配置模型"""
    provider: str
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 128000
    description: str = ""
    
class ExpertModelConfig(BaseModel):
    """专家模型配置"""
    model_id: str
    provider: str
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.5
    max_tokens: int = 2048
    description: str = ""
    capabilities: List[str] = Field(default_factory=list)
    speed: str = "medium"  # fast, medium, slow
    cost: str = "medium"   # low, medium, high
    best_for: List[str] = Field(default_factory=list)

class Settings:
    """全局配置管理器，完全基于环境变量"""

    def __init__(self):
        # API凭证 (关键配置，必须提供)
        self.NEWAPI_KEY = os.getenv("NEWAPI_KEY")
        if not self.NEWAPI_KEY:
            raise ValueError("错误: 环境变量 `NEWAPI_KEY` 未设置。请在 .env 文件中提供您的API密钥。")
            
        self.NEWAPI_BASE_URL = os.getenv("NEWAPI_BASE_URL", "https://newapi.thefool.chat/v1")

        # 日志级别
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        # Agent模型配置 (从环境变量加载)
        self.DECISION_AGENT_MODEL = os.getenv("DECISION_AGENT_MODEL", "gemini-2.5-pro-nothinking-VCAI")
        self.COORDINATION_AGENT_MODEL = os.getenv("COORDINATION_AGENT_MODEL", "gemini-2.5-pro-nothinking-VCAI")
        self.DEFAULT_EXPERT_MODEL = os.getenv("DEFAULT_EXPERT_MODEL", "glm-4.5")

        # 动态构建模型映射
        self.MODEL_MAPPINGS = self._create_model_mappings()
        self.EXPERT_MODELS = self._create_expert_models()

    def _create_model_mappings(self) -> Dict[str, Any]:
        """动态创建基础Agent的模型映射"""
        return {
            "decision_agent": {
                "provider": "openai",
                "model_name": self.DECISION_AGENT_MODEL,
                "base_url": self.NEWAPI_BASE_URL,
                "api_key": self.NEWAPI_KEY,
                "temperature": 0.1,
                "max_tokens": 65535,
                "description": "第一层决策Agent，需要最强推理能力"
            },
            "coordination_agent": {
                "provider": "openai",
                "model_name": self.COORDINATION_AGENT_MODEL,
                "base_url": self.NEWAPI_BASE_URL,
                "api_key": self.NEWAPI_KEY,
                "temperature": 0.3,
                "max_tokens": 65535,
                "description": "第二层协调Agent，优秀的理解和协调能力"
            }
        }

    def _create_expert_models(self) -> Dict[str, Any]:
        """动态创建专家模型的配置池"""
        # 模型配置可以进一步从JSON文件或更复杂的配置结构加载
        # 这里为了简化，我们只从环境变量更新关键部分
        
        # 模板
        expert_models_template = {
            "qwen_30b": { "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct-SF", "description": "Qwen 30B - 轻量级模型，适合简单任务", "capabilities": ["code", "general", "fast_response"], "speed": "fast", "cost": "low", "best_for": ["simple_queries", "code_snippets", "quick_answers"], "temperature": 0.7, "max_tokens": 128000 },
            "glm-4.5": { "model_name": "glm-4.5-official", "description": "glm-4.5-official - 速度快，适合中等复杂度任务", "capabilities": ["general", "search", "analysis"], "speed": "fast", "cost": "low", "best_for": ["search_tasks", "data_extraction", "summarization"], "temperature": 0.5, "max_tokens": 128000 },
            "gemini-2.5-pro": { "model_name": "gemini-2.5-pro-free1", "description": "gemini-2.5-pro - 稳定可靠，适合复杂推理", "capabilities": ["reasoning", "analysis", "comprehensive"], "speed": "medium", "cost": "medium", "best_for": ["complex_analysis", "multi_step_reasoning", "detailed_research"], "temperature": 0.4, "max_tokens": 128000 },
            "glm-4.5-airx": { "model_name": "glm-4.5-airx-official", "description": "glm-4.5-airx-official - 快速精准，适合结构化任务", "capabilities": ["structured_output", "fast_response", "analysis"], "speed": "fast", "cost": "low", "best_for": ["data_parsing", "structured_analysis", "quick_summaries"], "temperature": 0.3, "max_tokens": 128000 },
            "claude-opus-4": { "model_name": "claude-opus-4-20250514-VCAI", "description": "Claude Opus 4 - 强大的推理能力，适合深度分析", "capabilities": ["deep_reasoning", "code", "math", "comprehensive"], "speed": "slow", "cost": "high", "best_for": ["complex_problems", "deep_analysis", "technical_tasks"], "temperature": 0.3, "max_tokens": 128000 }
        }

        # 动态填充通用配置
        for model_id, config in expert_models_template.items():
            config["model_id"] = model_id
            config["provider"] = "openai"
            config["base_url"] = self.NEWAPI_BASE_URL
            config["api_key"] = self.NEWAPI_KEY
            
        return expert_models_template
    
    def get_llm_config(self, agent_type: str) -> LLMConfig:
        """获取指定Agent类型的LLM配置"""
        config_dict = self.MODEL_MAPPINGS.get(agent_type)
        if not config_dict:
            raise ValueError(f"未知的Agent类型: {agent_type}")
        return LLMConfig(**config_dict)

    def get_expert_model_config(self, model_id: str) -> ExpertModelConfig:
        """获取专家模型配置"""
        config_dict = self.EXPERT_MODELS.get(model_id)
        if not config_dict:
            # 使用默认模型
            config_dict = self.EXPERT_MODELS.get(self.DEFAULT_EXPERT_MODEL)
            if not config_dict:
                # 如果默认模型也不存在，使用第一个可用模型
                config_dict = next(iter(self.EXPERT_MODELS.values()))
        return ExpertModelConfig(**config_dict)

    def list_expert_models(self) -> Dict[str, ExpertModelConfig]:
        """列出所有专家模型"""
        return {
            model_id: ExpertModelConfig(**config)
            for model_id, config in self.EXPERT_MODELS.items()
        }

settings = Settings()