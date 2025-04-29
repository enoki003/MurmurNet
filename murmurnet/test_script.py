# テスト用スクリプト
# MurmurNetの各モジュールの動作確認用

from murmurnet.modules import agent_pool, blackboard, input_reception, output_agent, rag_retriever, summary_engine

def main():
    print("MurmurNet テスト開始")
    # 各モジュールの初期化テスト
    try:
        ap = agent_pool.AgentPool()
        print("agent_pool: OK")
    except Exception as e:
        print(f"agent_pool: NG ({e})")
    try:
        bb = blackboard.Blackboard()
        print("blackboard: OK")
    except Exception as e:
        print(f"blackboard: NG ({e})")
    try:
        ir = input_reception.InputReception()
        print("input_reception: OK")
    except Exception as e:
        print(f"input_reception: NG ({e})")
    try:
        oa = output_agent.OutputAgent()
        print("output_agent: OK")
    except Exception as e:
        print(f"output_agent: NG ({e})")
    try:
        rag = rag_retriever.RagRetriever()
        print("rag_retriever: OK")
    except Exception as e:
        print(f"rag_retriever: NG ({e})")
    try:
        se = summary_engine.SummaryEngine()
        print("summary_engine: OK")
    except Exception as e:
        print(f"summary_engine: NG ({e})")
    print("MurmurNet テスト終了")

if __name__ == "__main__":
    main()
