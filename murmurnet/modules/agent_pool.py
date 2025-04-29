# エージェントプール管理の雛形
class AgentPoolManager:
    def __init__(self, config, blackboard):
        self.config = config
        self.blackboard = blackboard
        # エージェントリスト（ダミー）
        self.agents = []

    def run_agents(self, blackboard):
        # 各エージェントを実行（ダミー実装）
        pass
