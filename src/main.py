import asyncio

from services.llm_service import LLMService
from agents.react_agent import ReactAgent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from tools._temp import ping

mcp_client = MultiServerMCPClient({
    "rag_faiss": {
        "url": "http://localhost:8000/mcp/",
        "transport": "streamable_http",
    }
})

llm = LLMService('deepseek/deepseek-chat-v3-0324:free')
agent = ReactAgent(llm, [ping, *asyncio.run(mcp_client.get_tools())])

async def main():
    print("CLI Chatbot: Hello! Type 'bye' to exit.")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("CLI Chatbot: Goodbye!")
            break

        response = await agent.ainvoke(user_input)
        print("Agent:", response)

if __name__ == "__main__":
    asyncio.run(main())
