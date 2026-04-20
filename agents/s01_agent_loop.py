#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import sys
import json
import subprocess

from dotenv import load_dotenv

from rich import print

load_dotenv(override=True)

# 将项目根目录加入 sys.path，以便导入 llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.router import call_llm_with_tools

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# OpenAI 格式的工具定义
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
}]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- 核心模式：一个 while 循环，不断调用工具直到模型停止 --
def agent_loop(messages: list):
    while True:
        # 构建带 system 消息的完整消息列表
        full_messages = [{"role": "system", "content": SYSTEM}] + messages

        # print('当前信息{}'.format(messages))
        # print('历史信息{}'.format(full_messages))

        assistant_msg, finish_reason = call_llm_with_tools(full_messages, TOOLS)

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

            if func_name == "bash":
                command = args.get("command", "")  # 获取content里面的参数
                print(f"\033[33m$ {command}\033[0m")
                output = run_bash(command)  # 执行命令
                print(output[:200])
            else:
                output = f"Error: Unknown tool '{func_name}'"

            # OpenAI 格式：tool 角色的消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": output,
            })


if __name__ == "__main__":
    # 历史信息
    history = []

    while True:
        try:
            query = input("请输入你想说的内容：")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", "", "quit"):
            break
        history.append({"role": "user", "content": query})

        # 调用 agent
        agent_loop(history)

        print(history)

        # 打印最后一条 assistant 回复
        if history and history[-1].get("role") == "assistant":
            content = history[-1].get("content")
            if content:
                print(content)
        print()

