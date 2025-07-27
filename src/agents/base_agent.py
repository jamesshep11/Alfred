
class BaseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = []  # placeholder for now
    
    def invoke(self, message: str) -> str:
        self._observe(message)
        response = self._decide_and_act()
        return self._reflect(response.strip())

    def _observe(self, message):
        self.memory.append({"role": "user", "content": message})

    def _decide_and_act(self) -> str:
        return self.llm.chat(self.memory)  # naive for now

    def _reflect(self, response) -> str:
        self.memory.append({"role": "assistant", "content": response})
        return response
