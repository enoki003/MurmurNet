# 黒板（共有メモリ）モジュール雛形
class Blackboard:
    def __init__(self, config):
        self.config = config
        self.memory = {}

    def write(self, key, value):
        self.memory[key] = value

    def read(self, key):
        return self.memory.get(key)
