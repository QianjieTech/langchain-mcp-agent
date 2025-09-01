# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装基础工具和Node.js
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 注意：Node.js 20已经自带npx，不需要单独安装

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install uv && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# 创建数据目录
RUN mkdir -p /app/data /app/logs

# 环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# 启动命令
CMD ["python", "-m", "langgraph_mcp_agent.cli", "server"]
