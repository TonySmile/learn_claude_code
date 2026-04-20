#!/usr/bin/env python3
"""
Venus LLM 客户端 - 最简版，直接调用公司免费的 Venus API
配置从 .env 文件读取
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ==================== 从 .env 读取配置 ====================
VENUS_CONFIG = {
    'model_url': os.getenv('VENUS_BASE_URL', 'http://v2.open.venus.oa.com/llmproxy/chat/completions'),
    'model_name': os.getenv('VENUS_MODEL_glm5', 'kimi-k2-instruct-local'),
    'api_key': os.getenv('VENUS_API_KEY_liangbo', ''),
    'auth_type': 'Bearer',
    'timeout': 3600,
    'max_retries': 3,
    'temperature': 0.0,
}


def call_venus(messages, model=None, config=None, temperature=None):
    """
    调用 Venus API，返回模型回复内容（字符串）

    Args:
        messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "你好"}]
        model: 可选，指定模型名称（如 'glm-5'、'kimi-k2-instruct-local'），传了就用传的，没传就用 config 里的默认值
        config: 可选，自定义配置（默认用 VENUS_CONFIG）
        temperature: 可选，覆盖配置中的温度参数

    Returns:
        str: 模型回复内容，失败返回 None
    """
    cfg = config or VENUS_CONFIG
    url = cfg['model_url']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"{cfg['auth_type']} {cfg['api_key']}",
    }
    body = {
        'model': model or cfg['model_name'],  # 优先用手动传入的 model，没传就用配置里的默认值
        'messages': messages,
        'temperature': temperature if temperature is not None else cfg['temperature'],
    }

    for attempt in range(cfg.get('max_retries', 3)):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=cfg.get('timeout', 3600))
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                if content and content.strip():
                    return content
            else:
                print(f"[Venus] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[Venus] 请求异常: {e}")

    return None


# ==================== 快速测试 ====================
if __name__ == "__main__":
    # result = call_venus([
    #     {"role": "system", "content": "你是一个有用的助手。"},
    #     {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    # ])
    # print(f"回复: {result}")


    # 测试.env文件
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print(os.getenv('VENUS_API_KEY_liangbo'))
    print(os.getenv('NAME'))
