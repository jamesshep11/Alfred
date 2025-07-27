from langchain.agents import tool

@tool
def ping() -> bool:
    """A sample tool to test ReAct tool calling."""
    return True