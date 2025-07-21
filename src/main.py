from services import llm_service as llm

def main():
    print("CLI Chatbot: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("CLI Chatbot: Goodbye!")
            break

        response = generate_response(user_input)
        print(f"CLI Chatbot: {response}")

def generate_response(message):
    response = llm.invoke(message)
    
    return response

if __name__ == "__main__":
    main()
