#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI モデルサーバー
~~~~~~~~~~~~~~~~~~~~~
外部APIサーバー方式による真の並列処理実現

複数のサーバーインスタンスを起動し、
非同期リクエストで並列処理を実現

作者: Yuhi Sonoki
"""

import asyncio
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time
import json
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger('MurmurNet.ModelServer')

# リクエスト/レスポンスモデル
class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: Optional[str] = None

class GenerationResponse(BaseModel):
    content: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class AgentRequest(BaseModel):
    agent_id: int
    role: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.7

class AgentResponse(BaseModel):
    agent_id: int
    role: str
    content: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

# FastAPIアプリケーション
app = FastAPI(title="MurmurNet Model Server", version="1.0.0")

# グローバルモデルインスタンス（プロセス毎）
model_instance = None

def initialize_model():
    """モデルの初期化"""
    global model_instance
    if model_instance is None:
        from MurmurNet.modules.model_factory import ModelFactory
        factory = ModelFactory()
        model_instance = factory.create_model()
        logger.info("モデルインスタンスを初期化しました")

@app.on_event("startup")
async def startup_event():
    """サーバー起動時の初期化"""
    initialize_model()

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """テキスト生成エンドポイント"""
    start_time = time.time()
    
    try:
        global model_instance
        if model_instance is None:
            initialize_model()
        
        # プロンプト構築
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
        else:
            full_prompt = request.prompt
        
        # 推論実行（ブロッキング処理を別スレッドで実行）
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: model_instance.generate(
                    prompt=full_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
            )
        
        execution_time = time.time() - start_time
        
        return GenerationResponse(
            content=response,
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"生成エラー: {str(e)}")
        
        return GenerationResponse(
            content="",
            execution_time=execution_time,
            success=False,
            error_message=str(e)
        )

@app.post("/agent", response_model=AgentResponse)
async def process_agent(request: AgentRequest):
    """エージェント処理エンドポイント"""
    start_time = time.time()
    
    try:
        global model_instance
        if model_instance is None:
            initialize_model()
        
        # プロンプト構築
        full_prompt = f"System: {request.system_prompt}\n\nUser: {request.user_prompt}"
        
        # 推論実行
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: model_instance.generate(
                    prompt=full_prompt,
                    temperature=request.temperature,
                    max_tokens=512
                )
            )
        
        execution_time = time.time() - start_time
        
        return AgentResponse(
            agent_id=request.agent_id,
            role=request.role,
            content=response,
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"エージェント {request.agent_id} エラー: {str(e)}")
        
        return AgentResponse(
            agent_id=request.agent_id,
            role=request.role,
            content="",
            execution_time=execution_time,
            success=False,
            error_message=str(e)
        )

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "model_loaded": model_instance is not None}

# クライアント側の並列処理クラス
class AsyncAgentPool:
    """
    非同期APIクライアントによる並列エージェント処理
    
    複数のAPIサーバーに並列リクエストを送信して
    真の並列処理を実現
    """
    
    def __init__(self, server_urls: List[str]):
        """
        初期化
        
        Args:
            server_urls: APIサーバーのURLリスト
        """
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        
        # 役割テンプレート
        self.role_templates = {
            "discussion": [
                {"role": "多角的視点AI", "system": "あなたは多角的思考のスペシャリストです。論点を多面的に分析して議論の全体像を示してください。", "temperature": 0.7},
                {"role": "批判的思考AI", "system": "あなたは批判的思考の専門家です。前提や論理に疑問を投げかけ、新たな視点を提供してください。", "temperature": 0.8},
                {"role": "実証主義AI", "system": "あなたはデータと証拠を重視する科学者です。事実に基づいた分析と検証可能な情報を提供してください。", "temperature": 0.6},
                {"role": "倫理的視点AI", "system": "あなたは倫理学者です。道徳的・倫理的観点から議論を分析し、価値判断の視点を提供してください。", "temperature": 0.7}
            ],
            "default": [
                {"role": "バランス型AI", "system": "あなたは総合的な分析ができるバランス型AIです。公平で多面的な視点から回答してください。", "temperature": 0.7},
                {"role": "専門知識AI", "system": "あなたは幅広い知識を持つ専門家です。正確でわかりやすい情報を提供してください。", "temperature": 0.6}
            ]
        }
        
        logger.info(f"AsyncAgentPool初期化: {self.num_servers}サーバー")
    
    async def send_agent_request(self, session: aiohttp.ClientSession, 
                                server_url: str, request_data: AgentRequest) -> AgentResponse:
        """
        単一エージェントリクエストの送信
        
        Args:
            session: aiohttp セッション
            server_url: サーバーURL
            request_data: リクエストデータ
            
        Returns:
            AgentResponse: レスポンス
        """
        try:
            async with session.post(
                f"{server_url}/agent",
                json=request_data.dict(),
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return AgentResponse(**data)
                else:
                    error_text = await response.text()
                    return AgentResponse(
                        agent_id=request_data.agent_id,
                        role=request_data.role,
                        content="",
                        execution_time=0.0,
                        success=False,
                        error_message=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            return AgentResponse(
                agent_id=request_data.agent_id,
                role=request_data.role,
                content="",
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def execute_parallel(self, prompt: str, num_agents: int = 4) -> List[AgentResponse]:
        """
        並列エージェント実行
        
        Args:
            prompt: ユーザープロンプト
            num_agents: エージェント数
            
        Returns:
            List[AgentResponse]: 実行結果
        """
        # 役割の準備
        roles = self.role_templates["default"]
        requests = []
        
        for i in range(num_agents):
            role_info = roles[i % len(roles)]
            request_data = AgentRequest(
                agent_id=i,
                role=role_info["role"],
                system_prompt=role_info["system"],
                user_prompt=prompt,
                temperature=role_info["temperature"]
            )
            requests.append(request_data)
        
        # 並列リクエスト実行
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, request_data in enumerate(requests):
                server_url = self.server_urls[i % self.num_servers]
                task = self.send_agent_request(session, server_url, request_data)
                tasks.append(task)
            
            logger.info(f"並列リクエスト開始: {num_agents}エージェント")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 例外処理
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(AgentResponse(
                        agent_id=i,
                        role=requests[i].role,
                        content="",
                        execution_time=0.0,
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            successful = [r for r in processed_results if r.success]
            logger.info(f"並列処理完了: {len(successful)}/{len(processed_results)} 成功")
            
            return processed_results

def start_server(port: int = 8000, host: str = "127.0.0.1"):
    """サーバーの起動"""
    uvicorn.run(app, host=host, port=port, log_level="info")

def start_multiple_servers(ports: List[int], host: str = "127.0.0.1"):
    """複数サーバーの起動"""
    processes = []
    
    for port in ports:
        process = mp.Process(target=start_server, args=(port, host))
        process.start()
        processes.append(process)
        logger.info(f"サーバー起動: http://{host}:{port}")
    
    return processes

# 使用例
async def test_async_agent_pool():
    """テスト用の非同期関数"""
    # テスト用のサーバーURL（実際には起動済みのサーバーを指定）
    server_urls = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
        "http://127.0.0.1:8003"
    ]
    
    pool = AsyncAgentPool(server_urls)
    
    test_prompt = "人工知能の倫理的な課題について議論してください"
    
    results = await pool.execute_parallel(test_prompt, num_agents=4)
    
    # 結果の表示
    for result in results:
        if result.success:
            print(f"✅ {result.role}: {result.content[:100]}...")
        else:
            print(f"❌ {result.role}: {result.error_message}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # サーバーモード
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        start_server(port)
    elif len(sys.argv) > 1 and sys.argv[1] == "servers":
        # マルチサーバーモード
        ports = [8000, 8001, 8002, 8003]
        start_multiple_servers(ports)
        
        # プロセス終了まで待機
        try:
            input("Press Enter to stop all servers...")
        except KeyboardInterrupt:
            pass
    else:
        # クライアントテストモード
        asyncio.run(test_async_agent_pool())
