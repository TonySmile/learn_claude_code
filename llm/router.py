#!/usr/bin/env python3
"""
LLM 路由模块 - 统一入口（支持前端选择模型）
==========================================

通过 .env 中的 LLM_PROVIDER 设置默认提供商，
同时支持调用时通过 model 参数动态切换模型（类似 ChatGPT 选模型）。

用法:
    from llm.router import call_llm, get_available_models

    # 方式1：使用默认模型（由 .env 决定）
    reply = call_llm([{"role": "user", "content": "你好"}])

    # 方式2：指定模型（前端选择）
    reply = call_llm([{"role": "user", "content": "你好"}], model="gpt-5")
    reply = call_llm([{"role": "user", "content": "你好"}], model="claude-3.5-sonnet")

    # 方式3：获取所有可用模型（供前端下拉框使用）
    models = get_available_models()
    # [{"model": "gpt-5", "provider": "timi"}, {"model": "kimi-k2-instruct-local", "provider": "venus"}, ...]

切换默认: 只需修改 .env 中的一行
    LLM_PROVIDER=venus   # 默认使用 Venus
    LLM_PROVIDER=timi    # 默认使用 TIMI AI HUB
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

from llm.venus_client import VENUS_CONFIG
from llm.timi_client import TIMI_CONFIG


# ==================== 模型注册表 ====================
# 每个模型直接映射到它的 provider config 和 provider 名称
# 前端展示的可选模型列表就从这里来
# 格式: 'model_name': {'provider': 'xxx', 'config': XXX_CONFIG}
# 想加新模型/新 provider，在这里加一行即可
MODEL_REGISTRY = {
    # ----- TIMI 可用模型（对应 .env 中 TIMI_MODEL_xxx）-----
    'claude-opus-4.6':          {'provider': 'timi',  'config': TIMI_CONFIG},
    'claude-opus-4.7':          {'provider': 'timi',  'config': TIMI_CONFIG},
    'gpt-5.4':                  {'provider': 'timi',  'config': TIMI_CONFIG},
    # ----- Venus 可用模型（对应 .env 中 VENUS_MODEL_xxx）-----
    'kimi-k2-instruct-local':   {'provider': 'venus', 'config': VENUS_CONFIG},
    'glm-4.7':                  {'provider': 'venus', 'config': VENUS_CONFIG},
    'glm-5':                    {'provider': 'venus', 'config': VENUS_CONFIG},
}


# ==================== 根据 .env 选择默认 Provider ====================
# PROVIDERS 仅用于 .env 默认配置查找，模型路由走 MODEL_REGISTRY
_PROVIDERS = {
    'venus': VENUS_CONFIG,
    'timi':  TIMI_CONFIG,
}

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'venus').strip().lower()

if LLM_PROVIDER not in _PROVIDERS:
    raise ValueError(
        f"未知的 LLM_PROVIDER='{LLM_PROVIDER}'，可选值: {list(_PROVIDERS.keys())}"
    )

# 默认配置对象 —— 不指定 model 时使用
LLM_CONFIG = _PROVIDERS[LLM_PROVIDER]

print(f"[LLM Router] 默认使用: {LLM_PROVIDER} | model={LLM_CONFIG['model_name']} | auth={LLM_CONFIG.get('auth_type', 'Bearer')}")
print(f"[LLM Router] 可选模型: {list(MODEL_REGISTRY.keys())}")


# ==================== 模型解析：根据 model 名找到对应配置 ====================
def resolve_model_config(model=None):
    """
    根据模型名解析出完整的调用配置。

    Args:
        model: 模型名（如 'gpt-5'、'claude-3.5-sonnet'），为 None 则用默认配置

    Returns:
        dict: 包含 model_url, model_name, api_key, auth_type 等的配置字典
    """
    if model is None:
        return LLM_CONFIG

    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的模型 '{model}'，可选值: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[model]
    # 用 provider 的基础配置，但覆盖 model_name 为用户指定的模型
    return {**entry['config'], 'model_name': model}


# ==================== 获取可用模型列表（供前端使用） ====================
def get_available_models():
    """
    返回所有可用模型列表，供前端下拉框展示。

    Returns:
        list[dict]: 如 [{"model": "gpt-5", "provider": "timi", "is_default": True}, ...]
    """
    default_model = LLM_CONFIG['model_name']
    models = []
    for model_name, entry in MODEL_REGISTRY.items():
        models.append({
            'model': model_name,
            'provider': entry['provider'],
            'is_default': (model_name == default_model),
        })
    return models


# ==================== 统一的认证头构造 ====================
def build_auth_header(cfg=None):
    """
    根据 auth_type 构造 Authorization header。
    - Bearer: 'Bearer xxx'   (Venus / OpenAI 标准)
    - Raw:    'xxx'          (TIMI AI HUB)
    """
    cfg = cfg or LLM_CONFIG
    auth_type = cfg.get('auth_type', 'Bearer')
    api_key = cfg.get('api_key', '')
    if auth_type == 'Raw':
        return api_key
    return f"{auth_type} {api_key}"


# ==================== 统一的 LLM 调用（带工具支持） ====================
def call_llm_with_tools(messages, tools=None, config=None, model=None):
    """
    统一的 LLM 调用（OpenAI 格式），支持工具调用和动态模型选择。

    Args:
        messages: OpenAI 格式的消息列表
        tools: 工具定义列表（可选）
        config: 自定义配置（可选，优先级最高）
        model: 模型名（可选，如 'gpt-5'、'claude-3.5-sonnet'）

    优先级: config > model > 默认配置

    Returns:
        (assistant_msg: dict, finish_reason: str)
        失败时返回 (None, 'error')
    """
    # 优先级：显式传入 config > 通过 model 解析 > 默认配置
    if config:
        cfg = config
    else:
        cfg = resolve_model_config(model)

    url = cfg['model_url']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': build_auth_header(cfg),
    }
    body = {
        'model': cfg['model_name'],
        'messages': messages,
        'temperature': cfg.get('temperature', 0.0),
    }
    if tools:
        body['tools'] = tools

    # 用模型名作为日志标签，更直观
    tag = cfg['model_name'].upper()
    for attempt in range(cfg.get('max_retries', 3)):
        try:
            resp = requests.post(
                url, headers=headers,
                data=json.dumps(body),
                timeout=cfg.get('timeout', 3600),
            )
            if resp.status_code == 200:
                data = resp.json()
                choice = data['choices'][0]
                return choice['message'], choice.get('finish_reason', 'stop')
            else:
                print(f"[{tag}] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[{tag}] 请求异常 (第{attempt + 1}次): {e}")

    return None, 'error'


# ==================== 简化版：不带工具的调用 ====================
def call_llm(messages, config=None, temperature=None, model=None):
    """
    简化版调用，直接返回回复内容字符串。

    Args:
        messages: OpenAI 格式的消息列表
        config: 自定义配置（可选）
        temperature: 温度参数（可选）
        model: 模型名（可选，如 'gpt-5'、'claude-3.5-sonnet'）

    Returns:
        str: 模型回复内容，失败返回 None
    """
    # 优先级：显式传入 config > 通过 model 解析 > 默认配置
    if config:
        cfg = config
    else:
        cfg = resolve_model_config(model)

    if temperature is not None:
        cfg = {**cfg, 'temperature': temperature}

    msg, _ = call_llm_with_tools(messages, tools=None, config=cfg)
    if msg is None:
        return None
    content = msg.get('content', '')
    return content if content and content.strip() else None


# ==================== 快速测试 ====================
if __name__ == "__main__":
    # 测试1：使用默认模型
    print("=== 测试默认模型 ===")
    reply = call_llm([
        {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    ])
    print(f"回复: {reply}")

    # 测试2：指定模型
    print("\n=== 测试指定模型 claude-opus-4.7 ===")
    reply = call_llm(
        [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
        model="claude-opus-4.7",
    )
    print(f"回复: {reply}")

    # 测试3：查看可用模型列表
    print("\n=== 可用模型列表 ===")
    for m in get_available_models():
        default_mark = " ⭐默认" if m['is_default'] else ""
        print(f"  {m['model']} ({m['provider']}){default_mark}")