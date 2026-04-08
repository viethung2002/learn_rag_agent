import asyncio

from fastmcp import Client


async def main():
    async with Client("http://127.0.0.1:8100/mcp") as client:
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])

        result = await client.call_tool(
            "route_query_via_llm",
            {
                "query": (
                    'Which shared citations do "Attention Is All You Need" and '
                    '"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" have?'
                )
            },
        )
        print("Result:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
