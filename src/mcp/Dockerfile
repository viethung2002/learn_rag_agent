FROM python:3.12-slim

WORKDIR /app/mcp

RUN pip install --no-cache-dir uv

# COPY src/mcp/pyproject.toml ./

# UV_COMPILE_BYTECODE for generating .pyc files -> faster application startup.
# UV_LINK_MODE=copy to silence warnings about not being able to use hard links
# since the cache and sync target are on separate file systems.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    # --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/mcp/pyproject.toml \
    uv sync --no-dev

# PYTHONUNBUFFERED=1 to disable output buffering
ENV PYTHONUNBUFFERED=1
ARG VERSION=0.1.0
ENV APP_VERSION=$VERSION

ENV PATH="/app/mcp/.venv/bin:$PATH"

EXPOSE 8100
CMD ["python", "-m", "src.mcp.mcp_server"]
