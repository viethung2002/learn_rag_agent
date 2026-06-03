FROM python:3.12-slim
WORKDIR /app/mcp

RUN pip install --no-cache-dir uv
RUN pip install langchain-nvidia-ai-endpoints
# Thiết lập môi trường uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Cài đặt dependencies từ toml vào .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/app/mcp/pyproject.toml \
    uv sync --no-dev

# Quan trọng: Copy toàn bộ code vào để chạy được module src.mcp
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/mcp/.venv/bin:$PATH"

EXPOSE 8100
CMD ["python", "-m", "src.mcp.mcp_server"]
