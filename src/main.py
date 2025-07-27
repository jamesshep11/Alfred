from services.llm_service import LLMService
from agents.base_agent import BaseAgent

llm = LLMService('google/gemma-3n-e2b-it:free')
agent = BaseAgent(llm)

def main():
    print("CLI Chatbot: Hello! Type 'bye' to exit.")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("CLI Chatbot: Goodbye!")
            break

        response = agent.invoke(user_input)
        print("Agent:", response)

if __name__ == "__main__":
    main()
