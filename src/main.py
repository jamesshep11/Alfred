from services.llm_service import LLMService
from agents.react_agent import ReactAgent
from tools._temp import ping

llm = LLMService('qwen/qwen3-coder:free')
agent = ReactAgent(llm, [ping])

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
