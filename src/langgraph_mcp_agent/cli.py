#src\langgraph_mcp_agent\cli.py
"""简化的CLI工具 - 专注于聊天交互"""
import asyncio
import httpx
import json
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from typing import Optional

console = Console()

API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 300  # 5分钟超时

class ChatClient:
    """聊天客户端"""
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = httpx.Timeout(timeout=timeout)
        self.session_id = None
    
    async def start_session(self) -> str:
        """开始新会话"""
        # 会话ID会在第一次请求时自动创建
        return "new_session"
    
    async def send_message(self, message: str, stream: bool = True):
        """发送消息"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if stream:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": message}],
                        "stream": True,
                        "session_id": self.session_id
                    }
                ) as response:
                    buffer = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        buffer += content
                                        yield content
                                
                                # 保存session_id
                                if not self.session_id and "id" in chunk:
                                    # 从completion id中提取session_id
                                    completion_id = chunk["id"]
                                    if completion_id.startswith("chatcmpl-"):
                                        self.session_id = completion_id[9:]
                            except json.JSONDecodeError:
                                pass
            else:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": message}],
                        "stream": False,
                        "session_id": self.session_id
                    }
                )
                result = response.json()
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    yield content
                
                # 保存session_id
                if not self.session_id and "id" in result:
                    completion_id = result["id"]
                    if completion_id.startswith("chatcmpl-"):
                        self.session_id = completion_id[9:]

@click.group()
def cli():
    """LangGraph MCP 聊天助手"""
    pass

@cli.command()
@click.option('--no-stream', is_flag=True, help='禁用流式输出')
def chat(no_stream):
    """交互式聊天模式"""
    asyncio.run(interactive_chat(stream=not no_stream))

async def interactive_chat(stream: bool = True):
    """交互式聊天"""
    console.print(Panel.fit(
        "[bold cyan]LangGraph MCP 智能助手[/bold cyan]\n"
        "[dim]输入 /exit 退出，/clear 清屏，/new 新会话[/dim]",
        border_style="cyan"
    ))
    
    client = ChatClient()
    
    while True:
        try:
            # 获取用户输入
            user_input = console.input("\n[bold green]你:[/bold green] ")
            
            # 处理命令
            if user_input.lower() == "/exit":
                console.print("[yellow]再见！[/yellow]")
                break
            elif user_input.lower() == "/clear":
                console.clear()
                continue
            elif user_input.lower() == "/new":
                client.session_id = None
                console.print("[yellow]开始新会话[/yellow]")
                continue
            elif not user_input.strip():
                continue
            
            # 发送消息并显示回复
            console.print("\n[bold blue]助手:[/bold blue]", end=" ")
            
            if stream:
                # 流式输出
                async for chunk in client.send_message(user_input, stream=True):
                    console.print(chunk, end="")
                console.print()  # 换行
            else:
                # 一次性输出
                async for content in client.send_message(user_input, stream=False):
                    console.print(content)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]使用 /exit 退出[/yellow]")
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")

@cli.command()
def server():
    """启动API服务器"""
    console.print("[cyan]启动 LangGraph MCP Chat API 服务器...[/cyan]")
    from langgraph_mcp_agent.api.server import run_server
    run_server()

@cli.command()
@click.argument('message')
@click.option('--stream/--no-stream', default=True, help='是否使用流式输出')
def ask(message, stream):
    """单次提问"""
    asyncio.run(single_query(message, stream))

async def single_query(message: str, stream: bool = True):
    """单次查询"""
    client = ChatClient()
    
    console.print(f"\n[bold green]问题:[/bold green] {message}")
    console.print("\n[bold blue]回答:[/bold blue]\n")
    
    try:
        async for chunk in client.send_message(message, stream=stream):
            console.print(chunk, end="")
        console.print()  # 换行
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")

@cli.command()
async def status():
    """查看系统状态"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/status")
            if response.status_code == 200:
                data = response.json()
                console.print(Panel.fit(
                    f"[bold]系统状态[/bold]\n\n"
                    f"初始化: {'✓' if data['initialized'] else '✗'}\n"
                    f"MCP服务器: {', '.join(data['available_mcp_servers'])}\n"
                    f"可用工具: {data['available_tools']}\n"
                    f"活跃会话: {data.get('active_sessions', 0)}",
                    border_style="green"
                ))
            else:
                console.print("[red]获取状态失败[/red]")
        except httpx.ConnectError:
            console.print("[red]无法连接到服务器，请先启动服务器[/red]")

if __name__ == "__main__":
    cli()