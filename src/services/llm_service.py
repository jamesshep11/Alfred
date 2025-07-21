import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

client = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPEN_ROUTER_API'),
    model='google/gemma-3n-e2b-it:free',
    temperature=0
)

def invoke(prompt: str) -> str:
    response = client.invoke(prompt)

    return response.content