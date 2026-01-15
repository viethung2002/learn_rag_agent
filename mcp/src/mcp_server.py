from fastmcp import FastMCP

mcp = FastMCP("arxiv-tools")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York2"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",   # hoặc "http" nếu version mới dùng tên này
        host="0.0.0.0",                # Quan trọng: bind tất cả interface (để Docker truy cập)
        port=8100,                     # Phải khớp với EXPOSE trong Dockerfile
        path="/mcp"                    # Đảm bảo endpoint là /mcp
    )
