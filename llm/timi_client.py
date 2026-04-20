#!/usr/bin/env python3
"""
TIMI AI HUB LLM 客户端 - 最简版，直接调用 TIMI AI HUB API
配置从 .env 文件读取
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ==================== 从 .env 读取配置 ====================
TIMI_CONFIG = {
    'model_url': os.getenv('TIMI_BASE_URL', 'http://api.timiai.woa.com/ai_api_manage/llmproxy/chat/completions'),
    'model_name': os.getenv('TIMI_MODEL_gpt-5.4', 'gpt-5.4'),
    'api_key': os.getenv('TIMI_API_KEY', ''),
    'auth_type': 'Raw',  # TIMI 的 Authorization 头直接传 api_key，无 Bearer 前缀
    'timeout': 3600,
    'max_retries': 3,
    'temperature': 0.0,
}


def call_timi(messages, model=None, config=None, temperature=None):
    """
    调用 TIMI AI HUB API，返回模型回复内容（字符串）

    Args:
        messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "你好"}]
        model: 可选，指定模型名称（如 'gpt-5.4'、'claude-opus-4.7'），传了就用传的，没传就用 config 里的默认值
        config: 可选，自定义配置（默认用 TIMI_CONFIG）
        temperature: 可选，覆盖配置中的温度参数

    Returns:
        str: 模型回复内容，失败返回 None
    """
    cfg = config or TIMI_CONFIG
    url = cfg['model_url']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': cfg['api_key'],
    }
    body = {
        'model': model or cfg['model_name'],  # 优先用手动传入的 model，没传就用配置里的默认值
        'messages': messages,
        'temperature': temperature if temperature is not None else cfg['temperature'],
    }

    for attempt in range(cfg.get('max_retries', 3)):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=cfg.get('timeout', 3600))
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                if content and content.strip():
                    return content
            else:
                print(f"[TIMI] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[TIMI] 请求异常 (第{attempt + 1}次): {e}")

    return None


# ==================== 快速测试 ====================
if __name__ == "__main__":
    result = call_timi(messages=[
        {"role": "user", "content": "你好，你知道你是什么模型么？"},
    ], model='claude-opus-4.7')
    print(f"回复: {result}")
