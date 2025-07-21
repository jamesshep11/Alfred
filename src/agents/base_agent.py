
class BaseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = []  # placeholder for now

    def observe(self, message):
        self.memory.append({"role": "user", "content": message})

    def decide_and_act(self):
        return self.llm.chat(self.memory)  # naive for now

    def reflect(self, response):
        self.memory.append({"role": "assistant", "content": response})
        return response
