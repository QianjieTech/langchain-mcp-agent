#src\langgraph_mcp_agent\core\data_processor.py
"""数据处理和清洗工具"""
import re
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import structlog

logger = structlog.get_logger()

class DataProcessor:
    """数据处理和清洗工具类"""
    
    @staticmethod
    def cleanup_llm_response(raw_response: str) -> str:
        """清理LLM响应中的格式标记"""
        if not raw_response:
            return ""
        
        # 移除代码块标记
        cleaned = re.sub(r'```[\w]*\n?', '', raw_response)
        cleaned = re.sub(r'```', '', cleaned)
        
        # 移除多余的空白
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取JSON对象"""
        if not text:
            return None
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试从代码块中提取
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试查找JSON对象
        brace_start = text.find('{')
        if brace_start != -1:
            brace_count = 0
            for i, char in enumerate(text[brace_start:], brace_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[brace_start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break
        
        return None
    
    @staticmethod
    def cleanup_mcp_response(raw_response: Any) -> Any:
        """清理MCP返回的原始数据"""
        if isinstance(raw_response, dict):
            cleaned = {}
            for key, value in raw_response.items():
                # 跳过系统字段
                if key.startswith('_') or key in ['metadata', 'schema', 'format', '__type__']:
                    continue
                
                # 递归清理
                if isinstance(value, (dict, list)):
                    cleaned[key] = DataProcessor.cleanup_mcp_response(value)
                else:
                    cleaned[key] = value
            return cleaned
        
        elif isinstance(raw_response, list):
            return [DataProcessor.cleanup_mcp_response(item) for item in raw_response]
        
        elif isinstance(raw_response, str):
            return DataProcessor.cleanup_llm_response(raw_response)
        
        else:
            return raw_response
    
    @staticmethod
    def merge_expert_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个专家的结果"""
        merged = {
            "combined_results": [],
            "successful_agents": [],
            "failed_agents": [],
            "tools_used": set(),
            "total_agents": len(results)
        }
        
        for result in results:
            if result.get("status") == "success":
                merged["successful_agents"].append(result.get("agent_id"))
                merged["combined_results"].append({
                    "agent_id": result.get("agent_id"),
                    "specialty": result.get("specialty"),
                    "result": result.get("result")
                })
                
                # 收集使用的工具
                tools = result.get("tools_used", [])
                if tools:
                    merged["tools_used"].update(tools)
            else:
                merged["failed_agents"].append({
                    "agent_id": result.get("agent_id"),
                    "error": result.get("error")
                })
        
        merged["tools_used"] = list(merged["tools_used"])
        merged["success_rate"] = len(merged["successful_agents"]) / merged["total_agents"] if merged["total_agents"] > 0 else 0
        
        return merged
    
    @staticmethod
    def format_for_output(data: Any, output_format: str = "text") -> str:
        """格式化输出数据"""
        if output_format == "json":
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        elif output_format == "markdown":
            if isinstance(data, dict):
                lines = []
                for key, value in data.items():
                    if isinstance(value, list):
                        lines.append(f"**{key}:**")
                        for item in value:
                            lines.append(f"- {item}")
                    else:
                        lines.append(f"**{key}:** {value}")
                return "\n".join(lines)
            else:
                return str(data)
        
        else:  # text
            if isinstance(data, dict):
                lines = []
                for key, value in data.items():
                    lines.append(f"{key}: {value}")
                return "\n".join(lines)
            else:
                return str(data)
    
    @staticmethod
    def calculate_quality_score(data: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        score = 0.0
        max_score = 10.0
        
        # 检查必要字段
        if data.get("result"):
            score += 2.0
        
        if data.get("status") == "success":
            score += 2.0
        
        # 检查结果完整性
        if isinstance(data.get("result"), str) and len(data["result"]) > 50:
            score += 2.0
        
        # 检查是否有错误
        if not data.get("error"):
            score += 2.0
        
        # 检查工具使用
        if data.get("tools_used"):
            score += 1.0
        
        # 检查执行时间
        if data.get("execution_time"):
            score += 1.0
        
        return min(score / max_score, 1.0)