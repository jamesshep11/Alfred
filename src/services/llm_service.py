import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

class LLMService():
    def __init__(self, model: str, temperature: float = 0):
        self.client = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPEN_ROUTER_API'),
            model=model,
            temperature=temperature
        )

    def prompt(self, message: str) -> str:
        """Send a standalone prompt"""
        response = self.client.invoke(message)

        return response.content

    def chat(self, messages: list) -> str:
        """Send a list of messages for contextual chats (chat history)"""
        response = self.client.invoke(messages)

        return response.content