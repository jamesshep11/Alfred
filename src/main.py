
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
    # Very dumb rule-based responses. Upgrade as needed.
    if "hello" in message.lower():
        return "Hey there!"
    elif "how are you" in message.lower():
        return "I'm just a script, but thanks for asking!"
    elif "joke" in message.lower():
        return "Why do programmers prefer dark mode? Because light attracts bugs."
    else:
        return "I'm not sure how to respond to that."

if __name__ == "__main__":
    main()
