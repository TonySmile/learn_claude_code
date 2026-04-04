#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

from rich import print

load_dotenv(override=True)

# 将项目根目录加入 sys.path，以便导入 llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.venus_client import VENUS_CONFIG

WORKDIR = Path.cwd()

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# 这是一个沙盒环境的保险机制，防止用户输入路径越界
def safe_path(p: str) -> Path:
    """Resolve a path relative to the workspace and check for escapes."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace")
        stdout = r.stdout or ""
        stderr = r.stderr or ""
        out = (stdout + stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# OpenAI 格式的工具定义 --> LLM看到就可以直接返回他觉得对的工具名称 & 参数 --> 我们只需要在里面加工具的描述和参数
# 那这里我自己思考：如果是skill的话我就有一个工具，渐进式将SKILL.md内容写入到Prompt当中 --> 提前和TOOLS一样先注册
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},

    {"type": "function", "function": {
        "name": "read_file", "description": "Read file contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},  # 这里只有path是必填

    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},

    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace exact text in file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]


def call_venus_with_tools(messages, tools=None):
    """
    调用 Venus API（OpenAI 格式），支持工具调用。
    返回完整的 response message 字典。
    """
    cfg = VENUS_CONFIG
    url = cfg['model_url']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"{cfg['auth_type']} {cfg['api_key']}",
    }
    body = {
        'model': cfg['model_name'],
        'messages': messages,
        'temperature': cfg['temperature'],
    }
    if tools:
        body['tools'] = tools

    for attempt in range(cfg.get('max_retries', 3)):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=cfg.get('timeout', 3600))
            if resp.status_code == 200:
                data = resp.json()
                choice = data['choices'][0]
                return choice['message'], choice.get('finish_reason', 'stop')
            else:
                print(f"[Venus] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[Venus] 请求异常: {e}")

    return None, 'error'


def agent_loop(messages: list):
    while True:
        # 构建带 system 消息的完整消息列表
        full_messages = [{"role": "system", "content": SYSTEM}] + messages

        # 调用LLM
        assistant_msg, finish_reason = call_venus_with_tools(full_messages, TOOLS)

        if assistant_msg is None:
            print("\033[31m[Error] API 调用失败\033[0m")
            return

        # 将 assistant 回复加入历史
        messages.append({"role": "assistant", "content": assistant_msg.get("content"),
                         "tool_calls": assistant_msg.get("tool_calls")})

        # 如果模型没有调用工具，结束循环
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            return

        # 执行每个工具调用，收集结果
        for tool_call in tool_calls:
            func = tool_call["function"]
            func_name = func["name"]
            try:
                args = json.loads(func["arguments"])
            except json.JSONDecodeError:
                args = {"command": func["arguments"]}

            handler = TOOL_HANDLERS.get(func_name)
            if handler:
                print(f"\033[33m> {func_name}: {json.dumps(args, ensure_ascii=False)[:200]}\033[0m")
                output = handler(**args)
                print(output[:200])
            else:
                output = f"Error: Unknown tool '{func_name}'"

            # OpenAI 格式：tool 角色的消息 --> 将工具调用的返回结果的值作为content，假如到历史信息 --> 遵循OpenAI格式
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": output,
            })


        # # 打印所有历史信息
        # print('#' * 100)
        # print('打印所有历史信息:')
        # print(messages)
        # print('#' * 100)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("请输入你的问题:")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 没调用工具，调用了则继续，就结束这个小循环， --> 纯粹的Agent循环


        # 打印最后一条 assistant 回复
        if history and history[-1].get("role") == "assistant":
            content = history[-1].get("content")
            if content:
                print(content)
        print()

        # 你能读取一下s01_agent_loop.py这个文件，并总结一下内容么？
        # 你能读取到s01_agent_loop.py这个文件里面的内容么？

