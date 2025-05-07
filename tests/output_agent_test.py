# OutputAgentの簡単なテスト
from output_agent import OutputAgent

class DummyBlackboard:
    prompt = "hello"

if __name__ == "__main__":
    config = {}
    agent = OutputAgent(config)
    blackboard = DummyBlackboard()
    response = agent.generate(blackboard)
    print(f"OutputAgent.generateの返り値: {response}")
