#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Fresh messages=[] gives context isolation. The parent stays clean."

Note: Real Claude Code also uses in-process isolation (not OS-level process
forking). The child runs in the same process with a fresh message array and
isolated tool context -- same pattern as this teaching implementation.

    Comparison with real Claude Code:
    +-------------------+------------------+----------------------------------+
    | Aspect            | This demo        | Real Claude Code                 |
    +-------------------+------------------+----------------------------------+
    | Backend           | in-process only  | 5 backends: in-process, tmux,    |
    |                   |                  | iTerm2, fork, remote             |
    | Context isolation | fresh messages=[]| createSubagentContext() isolates  |
    |                   |                  | ~20 fields (tools, permissions,  |
    |                   |                  | cwd, env, hooks, etc.)           |
    | Tool filtering    | manually curated | resolveAgentTools() filters from |
    |                   |                  | parent pool; allowedTools         |
    |                   |                  | replaces all allow rules         |
    | Agent definition  | hardcoded system | .claude/agents/*.md with YAML    |
    |                   | prompt           | frontmatter (AgentTemplate)      |
    +-------------------+------------------+----------------------------------+
"""

import os
import re
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


class AgentTemplate:
    """
    Parse agent definition from markdown frontmatter.

    Real Claude Code loads agent definitions from .claude/agents/*.md.
    Frontmatter fields: name, tools, disallowedTools, skills, hooks,
    model, effort, permissionMode, maxTurns, memory, isolation, color,
    background, initialPrompt, mcpServers.
    3 sources: built-in, custom (.claude/agents/), plugin-provided.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.stem
        self.config = {}
        self.system_prompt = ""
        self._parse()

    def _parse(self):
        text = self.path.read_text()
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            self.system_prompt = text
            return
        for line in match.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                self.config[k.strip()] = v.strip()
        self.system_prompt = match.group(2).strip()
        self.name = self.config.get("name", self.name)


# -- Tool implementations shared by parent and child --
def safe_path(p: str) -> Path:
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
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
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


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning) -- OpenAI 格式
CHILD_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},

    {"type": "function", "function": {
        "name": "read_file", "description": "Read file contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},

    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},

    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace exact text in file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]


def call_venus_with_tools(messages, system_prompt, tools=None):
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
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    body = {
        'model': cfg['model_name'],
        'messages': full_messages,
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


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        assistant_msg, finish_reason = call_venus_with_tools(sub_messages, SUBAGENT_SYSTEM, CHILD_TOOLS)

        if assistant_msg is None:
            return "(subagent API error)"

        # 将 assistant 回复加入历史
        sub_messages.append({"role": "assistant", "content": assistant_msg.get("content"),
                             "tool_calls": assistant_msg.get("tool_calls")})

        # 如果模型没有调用工具，结束循环
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            break

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
                print(f"\033[33m  [subagent] > {func_name}: {json.dumps(args, ensure_ascii=False)[:200]}\033[0m")
                output = handler(**args)
                print(f"  {str(output)[:200]}")
            else:
                output = f"Unknown tool: {func_name}"

            # OpenAI 格式：tool 角色的消息
            sub_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(output)[:50000],
            })

    # Only the final text returns to the parent -- child context is discarded
    last_assistant = sub_messages[-1] if sub_messages else {}
    if last_assistant.get("role") == "assistant" and last_assistant.get("content"):
        return last_assistant["content"]
    return "(no summary)"


# -- Parent tools: base tools + task dispatcher --
# 实际上这里就是：
# 父工具 = 子工具 + 一个task工具
PARENT_TOOLS = CHILD_TOOLS + [
    {"type": "function", "function": {
        "name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}}},
]


def agent_loop(messages: list):
    while True:
        assistant_msg, finish_reason = call_venus_with_tools(messages, SYSTEM, PARENT_TOOLS)

        if assistant_msg is None:
            print("\033[31m[Error] API 调用失败\033[0m")
            return

        # 将 assistant 回复加入历史
        messages.append({"role": "assistant", "content": assistant_msg.get("content"),
                         "tool_calls": assistant_msg.get("tool_calls")})

        # 如果模型没有调用工具，结束循环
        tool_calls = assistant_msg.ge ("tool_calls")
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

            if func_name == "task":
                desc = args.get("description", "subtask")
                prompt = args.get("prompt", "")
                print(f"> task ({desc}): {prompt[:80]}")
                output = run_subagent(prompt)
            else:
                handler = TOOL_HANDLERS.get(func_name)
                if handler:
                    print(f"\033[33m> {func_name}: {json.dumps(args, ensure_ascii=False)[:200]}\033[0m")
                    output = handler(**args)
                else:
                    output = f"Unknown tool: {func_name}"

            print(f"  {str(output)[:200]}")

            # OpenAI 格式：tool 角色的消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(output),
            })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印最后一条 assistant 回复
        if history and history[-1].get("role") == "assistant":
            content = history[-1].get("content")
            if content:
                print(content)
        print()
