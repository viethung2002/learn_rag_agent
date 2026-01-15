# mcp_client.py
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8100/mcp") as client:
        tools = await client.list_tools()
        print("Available tools:", tools)

        result = await client.call_tool(
            "get_weather",
            {"location": "New York"}
        )
        print("Result:", result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
