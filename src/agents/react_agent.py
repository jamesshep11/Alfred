from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import InMemorySaver

class ReactAgent():
    def __init__(self, llm, tools: list[callable]):
        agent_tools = [Tool.from_function(tool, tool.name, tool.__doc__) for tool in tools]
        chat_history = InMemorySaver()
        
        self.agent = create_react_agent(
            model = llm.client,
            tools=agent_tools,
            checkpointer=chat_history
        )
    
    def invoke(self, message: str) -> str:
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config = {"configurable": {"thread_id": "1"}}
            )
        return response.get('messages')[-1].content.strip()