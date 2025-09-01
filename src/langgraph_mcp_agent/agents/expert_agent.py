# src/langgraph_mcp_agent/agents/expert_agent.py
"""强制使用MCP的专家Agent模块"""
import json
import re
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
import structlog

logger = structlog.get_logger()

class MCPMandatoryExpertAgent:
    """强制使用MCP工具的专家Agent - 修复动态模型选择"""
    
    def __init__(self, llm_manager, config: Dict[str, Any]):
        self.llm_manager = llm_manager
        self.agent_id = config.get("agent_id", "expert")
        self.specialty = config.get("specialty", "通用专家")
        self.task_description = config.get("task_description", "")
        self.all_tools = config.get("all_tools", [])
        self.mandatory_tools = config.get("mandatory_tools", [])
        self.original_query = config.get("original_query", "")
        self.complexity = config.get("complexity", "medium")
        self.model_preference = config.get("model_preference", "auto")
        
        # 修复：使用 assigned_tools（来自协调Agent）
        self.assigned_tools = config.get("assigned_tools", [])
        
        # 动态选择模型
        self.llm = self._select_model()
        self.required_tools = self._filter_mandatory_tools()
        self.react_agent = self._create_react_agent()
        
        # 记录实际使用的模型
        logger.info(f"专家 {self.agent_id} 初始化完成", 
                   specialty=self.specialty,
                   complexity=self.complexity,
                   model_info=self._get_model_info())
    
    def _select_model(self):
        """根据任务选择合适的模型"""
        # 如果有明确的模型偏好
        if self.model_preference != "auto" and self.model_preference in ["easy", "medium", "hard"]:
            task_type = self.specialty.lower()
            logger.info(f"专家 {self.agent_id} 使用指定复杂度: {self.model_preference}")
            return self.llm_manager.get_expert_model(task_type, self.model_preference)
        
        # 自动选择基于复杂度和专长
        task_type = self.specialty.lower()
        
        # 记录选择的模型信息
        model_id = self.llm_manager._select_expert_model(task_type, self.complexity)
        model_info = self.llm_manager.get_model_info(model_id)
        
        if model_info:
            logger.info(f"专家 {self.agent_id} 自动选择模型", 
                       model_id=model_id,
                       model_name=model_info.get('model_name'),
                       description=model_info.get('description'),
                       task_type=task_type,
                       complexity=self.complexity)
        
        return self.llm_manager.get_expert_model(task_type, self.complexity)
    
    def _filter_mandatory_tools(self) -> List:
        """筛选必须使用的MCP工具"""
        filtered = []
        
        for tool_name in self.mandatory_tools:
            for tool in self.all_tools:
                if (tool_name.lower() in tool.name.lower() or 
                    tool.name.lower() in tool_name.lower()):
                    if tool not in filtered:
                        filtered.append(tool)
                        logger.info(f"专家 {self.agent_id} 绑定工具: {tool.name}")
        
        return filtered
    
    def _create_react_agent(self):
        """创建React Agent"""
        try:
            tools_to_use = self.required_tools if self.required_tools else self.all_tools
            
            if not tools_to_use:
                logger.warning(f"专家 {self.agent_id} 没有可用工具")
                return None
            
            agent = create_react_agent(
                self.llm,
                tools=tools_to_use,
                state_modifier=self._create_system_prompt()
            )
            
            logger.info(
                f"专家 {self.agent_id} 创建React Agent成功",
                tools_count=len(tools_to_use),
                required_tools=[t.name for t in self.required_tools]
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"创建React Agent失败: {e}")
            return None
    
    def _create_system_prompt(self) -> str:
        """创建强制MCP调用的系统提示词"""
        tools_to_list = self.required_tools if self.required_tools else self.all_tools[:5]
        mandatory_tools_str = ", ".join([t.name for t in self.required_tools]) if self.required_tools else "可用工具"
        
        has_context7 = any('resolve-library-id' in t.name or 'get-library-docs' in t.name 
                           for t in self.required_tools)
        
        context7_instruction = ""
        if has_context7:
            context7_instruction = """
【Context7特殊要求】
你必须按以下顺序使用Context7工具：
1. 首先调用 resolve-library-id 来获取库ID
   - 参数: libraryName (库的名称，如 "rust", "react", "prisma" 等)
2. 使用返回的库ID调用 get-library-docs
   - 参数: context7CompatibleLibraryID (从第一步获取的ID，格式如 /org/project)
   
不要跳过第一步！必须先获取有效的库ID。
"""
        
        tools_desc = []
        for tool in tools_to_list[:10]:
            try:
                desc = getattr(tool, 'description', str(tool))[:100]
                tools_desc.append(f"- {tool.name}: {desc}")
            except:
                tools_desc.append(f"- {tool.name}")
        
        tools_list_str = "\n".join(tools_desc)
        
        # 根据复杂度调整提示词
        complexity_instruction = ""
        if self.complexity == "hard":
            complexity_instruction = """
【深度执行要求】
这是一个困难任务，你需要：
1. 多次调用MCP工具，从不同角度获取信息
2. 交叉验证不同来源的数据
3. 提供详细、全面的分析
4. 考虑边缘情况和特殊场景
"""
        elif self.complexity == "easy":
            complexity_instruction = """
【快速执行要求】
这是一个简单任务，你需要：
1. 快速调用相关MCP工具
2. 获取关键信息后立即总结
3. 保持回答简洁明了
"""
        
        return f"""你是{self.specialty}专家。

【强制规则】
你必须使用MCP工具来完成任务，禁止仅凭自己的知识回答。

任务: {self.task_description}
原始问题: {self.original_query}
任务复杂度: {self.complexity}

【必须使用的MCP工具】
{mandatory_tools_str}
{context7_instruction}

【可用工具示例】
{tools_list_str}

{complexity_instruction}

【执行要求】
1. 你必须调用MCP工具获取信息
2. 如果是搜索工具，使用合适的关键词进行搜索
3. 如果是Context7工具，必须先resolve-library-id再get-library-docs
4. 对工具返回的数据进行分析和概括
5. 明确说明你使用了哪个MCP工具

记住：必须使用工具，不允许直接回答！"""
    
    async def execute_with_mandatory_mcp(self, original_query: str, full_query: str) -> Dict[str, Any]:
        """执行任务（强制使用指定MCP）"""
        try:
            logger.info(f"专家 {self.agent_id} 开始执行 - 必须使用: {[t.name for t in self.required_tools]}")
            
            # 检查是否是Context7相关任务
            is_context7_task = any('resolve-library-id' in t.name or 'get-library-docs' in t.name 
                                  for t in self.required_tools)
            
            if is_context7_task:
                # Context7需要特殊处理
                return await self._execute_context7_workflow_fixed(original_query)
            
            if not self.react_agent:
                if self.required_tools:
                    return await self._force_tool_execution(original_query)
                return self._create_error_response("未配置必需的MCP工具")
            
            # 构建针对性查询
            if "search" in self.agent_id.lower():
                query = f"""
使用 {self.specialty} 搜索以下内容：
{original_query}

注意：你必须使用 {[t.name for t in self.required_tools]} 工具进行搜索。
"""
            else:
                query = f"""
{full_query}

使用 {[t.name for t in self.required_tools] if self.required_tools else '可用的MCP工具'} 来获取信息并回答。
"""
            
            # 执行任务
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = await self.react_agent.ainvoke({
                        "messages": [HumanMessage(content=query)]
                    })
                    
                    tools_used = self._extract_tools_used(result)
                    
                    if self.required_tools:
                        required_tool_names = {t.name for t in self.required_tools}
                        tool_used_correctly = any(tool in required_tool_names for tool in tools_used)
                    else:
                        tool_used_correctly = len(tools_used) > 0
                    
                    if tool_used_correctly:
                        final_answer = self._extract_final_answer(result)
                        
                        return {
                            "agent_id": self.agent_id,
                            "specialty": self.specialty,
                            "status": "success",
                            "summary": await self._create_summary(final_answer, tools_used),
                            "result": final_answer,
                            "tools_used": tools_used,
                            "task_description": self.task_description,
                            "model_used": self._get_model_info()
                        }
                    
                    logger.warning(f"专家 {self.agent_id} 第{attempt+1}次未使用必需工具")
                    
                except Exception as e:
                    logger.warning(f"专家 {self.agent_id} 第{attempt+1}次执行失败: {e}")
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，尝试强制调用
                        return await self._force_tool_execution(original_query)
            
            # 如果还是没用，强制调用
            return await self._force_tool_execution(original_query)
            
        except Exception as e:
            logger.error(f"专家 {self.agent_id} 执行失败: {e}")
            return self._create_error_response(str(e))
    
    def _get_model_info(self) -> Dict[str, str]:
        """获取当前使用的模型信息"""
        model_id = self.llm_manager._select_expert_model(
            self.specialty.lower(), 
            self.complexity
        )
        model_info = self.llm_manager.get_model_info(model_id)
        return {
            "model_id": model_id,
            "model_name": model_info.get("model_name", "unknown"),
            "description": model_info.get("description", "")
        }
    
    # 保留其他所有方法不变...
    async def _execute_context7_workflow_fixed(self, query: str) -> Dict[str, Any]:
        """执行Context7工作流 - 最终修复版本"""
        try:
            tools_used = []
            
            # 第一步：resolve-library-id
            resolve_tool = next((t for t in self.all_tools if 'resolve-library-id' in t.name), None)
            if not resolve_tool:
                return self._create_error_response("找不到resolve-library-id工具")
            
            # 提取库名
            library_name = self._extract_library_name(query)
            logger.info(f"Context7第一步：调用 {resolve_tool.name}，库名: {library_name}")
            
            # 调用resolve-library-id
            resolve_result = None
            error_messages = []
            
            # 尝试不同的参数格式
            params_to_try = [
                {"libraryName": library_name},
                {"name": library_name},
                {"query": library_name},
                {"input": library_name}
            ]
            
            for params in params_to_try:
                try:
                    logger.debug(f"尝试参数: {params}")
                    resolve_result = await self._direct_tool_invoke(resolve_tool, params)
                    if resolve_result:
                        logger.info(f"resolve-library-id成功返回: {str(resolve_result)[:200]}")
                        break
                except Exception as e:
                    error_msg = f"参数{params}失败: {str(e)}"
                    error_messages.append(error_msg)
                    logger.debug(error_msg)
            
            if not resolve_result:
                return self._create_error_response(
                    f"resolve-library-id调用失败，尝试的所有参数均失败:\n" + "\n".join(error_messages)
                )
            
            tools_used.append(resolve_tool.name)
            
            # 解析库ID - 更严格的解析
            library_id = self._parse_library_id_strict(resolve_result, library_name)
            
            if not library_id:
                # 如果没有找到有效的库ID，返回resolve结果并说明问题
                return {
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "status": "partial",
                    "summary": f"[Context7] 未找到'{library_name}'的有效库ID\n"
                              f"resolve-library-id返回内容:\n{str(resolve_result)[:500]}\n\n"
                              f"可能原因：\n"
                              f"1. 库名拼写错误或不存在\n"
                              f"2. Context7暂未索引该库\n"
                              f"3. 网络连接问题",
                    "result": str(resolve_result),
                    "tools_used": tools_used,
                    "task_description": self.task_description,
                    "model_used": self._get_model_info()
                }
            
            # 第二步：get-library-docs
            docs_tool = next((t for t in self.all_tools if 'get-library-docs' in t.name), None)
            if not docs_tool:
                return self._create_error_response("找不到get-library-docs工具")
            
            logger.info(f"Context7第二步：调用 {docs_tool.name} with ID: {library_id}")
            
            # 调用get-library-docs
            docs_result = None
            docs_params_to_try = [
                {"context7CompatibleLibraryID": library_id, "tokens": 5000},
                {"context7CompatibleLibraryID": library_id},
                {"library_id": library_id, "tokens": 5000},
                {"library_id": library_id},
                {"id": library_id}
            ]
            
            for params in docs_params_to_try:
                try:
                    logger.debug(f"尝试文档获取参数: {params}")
                    docs_result = await self._direct_tool_invoke(docs_tool, params)
                    if docs_result:
                        logger.info(f"get-library-docs返回内容长度: {len(str(docs_result))}")
                        break
                except Exception as e:
                    logger.debug(f"文档获取参数{params}失败: {e}")
            
            # 检查文档获取结果 - 修复判断逻辑
            if not docs_result:
                return {
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "status": "partial",
                    "summary": f"[Context7] 文档获取无返回\n库ID: {library_id}",
                    "result": f"库ID: {library_id}",
                    "tools_used": tools_used,
                    "task_description": self.task_description,
                    "model_used": self._get_model_info()
                }
            
            tools_used.append(docs_tool.name)
            
            # 判断是否是真正的错误（更智能的判断）
            docs_str = str(docs_result)
            is_real_error = self._is_context7_error(docs_str)
            
            if is_real_error:
                # 真正的错误
                return {
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "status": "partial",
                    "summary": f"[Context7] 文档获取遇到错误\n"
                              f"库ID: {library_id}\n"
                              f"错误信息: {docs_str[:300]}\n\n"
                              f"注意：这可能是网络问题或服务暂时不可用",
                    "result": docs_str,
                    "tools_used": tools_used,
                    "task_description": self.task_description,
                    "model_used": self._get_model_info()
                }
            
            # 成功获取文档
            # 解析文档内容
            doc_summary = self._parse_context7_docs(docs_str)
            
            return {
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "status": "success",
                "summary": f"[Context7工作流成功]\n"
                          f"库名: {library_name}\n"
                          f"库ID: {library_id}\n"
                          f"{doc_summary}",
                "result": docs_str,
                "tools_used": tools_used,
                "task_description": self.task_description,
                "model_used": self._get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Context7工作流执行失败: {e}", exc_info=True)
            return self._create_error_response(f"Context7执行失败: {str(e)}")
    
    def _is_context7_error(self, response: str) -> bool:
        """判断Context7响应是否是真正的错误"""
        # 真正的错误标志
        real_error_indicators = [
            "Failed to fetch documentation",
            "Error fetching library documentation",
            "fetch failed",
            "404",
            "500",
            "TypeError:",
            "Error code:",
            "Please try again later"
        ]
        
        # 正常内容标志（即使包含Error关键词）
        success_indicators = [
            "CODE SNIPPETS",
            "TITLE:",
            "DESCRIPTION:",
            "SOURCE:",
            "========================",
            "github.com",
            "```",
            "function",
            "class",
            "import",
            "export",
            "const",
            "let",
            "var",
            "def",
            "fn",
            "pub",
            "impl"
        ]
        
        response_lower = response.lower()
        
        # 先检查是否包含成功标志
        has_success_indicator = any(indicator.lower() in response_lower for indicator in success_indicators)
        if has_success_indicator:
            logger.debug("Context7响应包含成功标志，判断为成功")
            return False
        
        # 再检查是否包含真正的错误标志
        has_real_error = any(indicator.lower() in response_lower for indicator in real_error_indicators)
        if has_real_error:
            # 双重检查：如果同时包含代码内容，可能是文档中的错误示例
            if "code" in response_lower or "snippet" in response_lower:
                logger.debug("Context7响应包含错误标志但也包含代码，判断为成功")
                return False
            logger.debug("Context7响应包含真正的错误标志")
            return True
        
        # 默认认为成功
        return False
    
    def _parse_context7_docs(self, docs_str: str) -> str:
        """解析Context7文档内容，生成摘要"""
        lines = docs_str.split('\n')
        summary_parts = []
        
        # 提取关键信息
        titles = []
        descriptions = []
        sources = []
        
        for i, line in enumerate(lines):
            if "TITLE:" in line:
                titles.append(line.replace("TITLE:", "").strip())
            elif "DESCRIPTION:" in line:
                descriptions.append(line.replace("DESCRIPTION:", "").strip())
            elif "SOURCE:" in line:
                sources.append(line.replace("SOURCE:", "").strip())
        
        # 构建摘要
        if titles:
            summary_parts.append(f"找到 {len(titles)} 个相关文档片段:")
            for i, (title, desc) in enumerate(zip(titles[:3], descriptions[:3]), 1):
                summary_parts.append(f"  {i}. {title}")
                if desc:
                    summary_parts.append(f"     {desc[:100]}...")
        else:
            # 如果没有明确的标题，尝试提取其他信息
            content_preview = docs_str[:500]
            if "CODE SNIPPETS" in content_preview:
                summary_parts.append("获取到代码片段和文档")
            else:
                summary_parts.append(f"文档内容预览：\n{content_preview}")
        
        return "\n".join(summary_parts)
    
    def _parse_library_id_strict(self, result: Any, library_name: str) -> Optional[str]:
        """从resolve-library-id结果中严格解析库ID"""
        try:
            result_str = str(result)
            logger.debug(f"解析库ID，原始结果: {result_str[:500]}")
            
            # 不要返回默认值 /org/project
            if "/org/project" in result_str and library_name not in result_str:
                logger.warning("检测到默认占位符/org/project，忽略")
                return None
            
            # 查找真实的库ID格式（例如 /rust-lang/rust, /facebook/react等）
            # 更严格的正则表达式，确保不是占位符
            pattern = r'/[\w-]+/[\w-]+(?:/[\w.-]+)?'
            matches = re.findall(pattern, result_str)
            
            # 过滤并优先选择最匹配的
            valid_matches = []
            best_matches = []  # 完全匹配库名的
            partial_matches = []  # 部分匹配的
            
            for match in matches:
                if match == "/org/project":
                    continue
                    
                match_lower = match.lower()
                library_lower = library_name.lower()
                
                # 完全匹配
                if f"/{library_lower}" in match_lower or f"{library_lower}/" in match_lower:
                    best_matches.append(match)
                    logger.info(f"找到最佳匹配的库ID: {match}")
                # 部分匹配
                elif library_lower in match_lower or any(part in match_lower for part in library_lower.split('-')):
                    partial_matches.append(match)
                    logger.info(f"找到部分匹配的库ID: {match}")
            
            # 优先返回最佳匹配
            if best_matches:
                # 对于rust，优先选择/rust-lang/rust
                if library_name.lower() == "rust":
                    for match in best_matches:
                        if match == "/rust-lang/rust":
                            return match
                return best_matches[0]
            
            if partial_matches:
                return partial_matches[0]
            
            # 尝试从JSON格式解析
            if isinstance(result, dict):
                # 直接查找包含库名的ID
                for key, value in result.items():
                    if isinstance(value, str) and '/' in value:
                        if library_name.lower() in value.lower():
                            if re.match(pattern, value) and value != "/org/project":
                                logger.info(f"从字典中找到库ID: {value}")
                                return value
                
                # 查找标准字段
                for field in ['library_id', 'id', 'libraryId', 'context7CompatibleLibraryID']:
                    if field in result:
                        value = result[field]
                        if value and value != "/org/project" and '/' in str(value):
                            logger.info(f"从字段{field}找到库ID: {value}")
                            return value
            
            # 尝试解析列表
            if isinstance(result, list) and result:
                for item in result:
                    if isinstance(item, dict):
                        # 递归解析
                        parsed = self._parse_library_id_strict(item, library_name)
                        if parsed:
                            return parsed
                    elif isinstance(item, str) and '/' in item:
                        if library_name.lower() in item.lower() and item != "/org/project":
                            if re.match(pattern, item):
                                logger.info(f"从列表中找到库ID: {item}")
                                return item
            
            logger.warning(f"无法解析出有效的库ID for '{library_name}'")
            return None
            
        except Exception as e:
            logger.error(f"解析库ID失败: {e}")
            return None
    
    async def _direct_tool_invoke(self, tool, params: Dict) -> Any:
        """直接调用工具，避免TaskGroup错误"""
        try:
            # 创建新的任务
            result = await tool.ainvoke(params)
            
            # 记录调用结果
            logger.debug(f"工具{tool.name}调用成功，参数: {params}")
            
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"工具调用超时: {tool.name}")
        except Exception as e:
            logger.debug(f"工具{tool.name}调用失败: {e}")
            raise e
    
    async def _force_tool_execution(self, query: str) -> Dict[str, Any]:
        """强制执行工具调用"""
        try:
            # 检查是否是Context7相关工具
            context7_tools = [t for t in self.required_tools 
                            if 'resolve-library-id' in t.name or 'get-library-docs' in t.name]
            
            if context7_tools:
                return await self._execute_context7_workflow_fixed(query)
            
            tools_to_try = self.required_tools if self.required_tools else self.all_tools[:3]
            
            if tools_to_try:
                tool = tools_to_try[0]
                logger.info(f"强制调用工具: {tool.name}")
                
                # 调用工具
                result = await self._invoke_tool_with_params(tool, query)
                
                return {
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "status": "forced",
                    "summary": f"[强制调用 {tool.name}]\n{str(result)[:500]}",
                    "result": str(result),
                    "tools_used": [tool.name],
                    "task_description": self.task_description,
                    "model_used": self._get_model_info()
                }
        except Exception as e:
            logger.error(f"强制工具调用失败: {e}")
        
        return self._create_error_response("无法执行MCP工具")
    
    async def _invoke_tool_with_params(self, tool, query: str) -> Any:
        """调用工具（智能参数匹配）"""
        tool_name = tool.name.lower()
        
        params_to_try = []
        
        # 根据工具名称匹配参数
        if "search" in tool_name or "tavily" in tool_name:
            params_to_try = [{"query": query}]
        elif "bing" in tool_name:
            params_to_try = [{"query": query}]
        elif "fetch" in tool_name:
            params_to_try = [{"url": query}]
        elif "time" in tool_name:
            params_to_try = [{}]
        elif "arxiv" in tool_name:
            params_to_try = [{"query": query}]
        elif "resolve-library-id" in tool_name:
            params_to_try = [{"libraryName": query}, {"name": query}]
        elif "get-library-docs" in tool_name:
            return "get-library-docs需要先调用resolve-library-id获取库ID"
        else:
            params_to_try = [{"input": query}, {"question": query}]
        
        for params in params_to_try:
            try:
                return await self._direct_tool_invoke(tool, params)
            except Exception as e:
                logger.debug(f"参数 {params} 失败: {e}")
                continue
        
        return f"工具调用失败: 所有参数尝试均失败"
    
    def _extract_library_name(self, query: str) -> str:
        """从查询中提取库名 - 改进版本"""
        # 对于Rust查询，返回rust
        if 'rust' in query.lower():
            return 'rust'
        
        # 查找引号中的内容
        quoted = re.findall(r'"([^"]*)"', query)
        if quoted:
            return quoted[0]
        
        quoted = re.findall(r"'([^']*)'", query)
        if quoted:
            return quoted[0]
        
        # 常见的库名关键词
        keywords = ['rust', 'react', 'vue', 'angular', 'express', 'django', 'flask', 
                   'tensorflow', 'pytorch', 'pandas', 'numpy', 'prisma', 
                   'nextjs', 'next.js', 'typescript', 'javascript', 'python']
        
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower:
                return keyword
        
        # 尝试提取技术相关词汇
        tech_pattern = r'\b(?:rust|react|vue|angular|express|django|flask|tensorflow|pytorch|pandas|numpy|prisma|nextjs|next\.js|typescript|javascript|python)\b'
        tech_match = re.search(tech_pattern, query, re.IGNORECASE)
        if tech_match:
            return tech_match.group()
        
        # 默认返回查询的核心词
        words = query.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                return word.lower()
        
        return "unknown"
    
    async def _create_summary(self, content: str, tools_used: List[str]) -> str:
        """创建概括"""
        tools_str = f"[使用MCP: {', '.join(tools_used)}]\n" if tools_used else ""
        
        if len(content) > 800:
            messages = [
                SystemMessage(content="概括以下内容的关键信息："),
                HumanMessage(content=content[:1500])
            ]
            response = await self.llm.ainvoke(messages)
            return tools_str + response.content
        
        return tools_str + content
    
    def _extract_tools_used(self, result: Dict) -> List[str]:
        """提取使用的工具列表"""
        tools = []
        for msg in result.get("messages", []):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    if tool_name not in tools:
                        tools.append(tool_name)
        return tools
    
    def _extract_final_answer(self, result: Dict) -> str:
        """提取最终答案"""
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    return msg.content
        return "未能生成有效回答"
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "status": "error",
            "summary": f"执行失败: {error}",
            "result": "",
            "tools_used": [],
            "task_description": self.task_description,
            "model_used": self._get_model_info()
        }