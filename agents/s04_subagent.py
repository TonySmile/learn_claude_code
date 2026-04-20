#!/usr/bin/env python3
# 核心机制：上下文隔离 -- 保护模型的思维清晰度
"""
s04_subagent.py - 子代理

生成一个拥有全新 messages=[] 的子代理。子代理在自己的上下文中工作，
与父代理共享文件系统，最终只返回摘要给父代理。

    父代理 (Parent)                   子代理 (Subagent)
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- 全新上下文
    |                  |  分发任务    |                  |
    | tool: task       | ---------->| while 工具调用:   |
    |   prompt="..."   |            |   调用工具        |
    |   description="" |            |   追加结果        |
    |                  |  返回摘要   |                  |
    |   result = "..." | <--------- | 返回最终文本      |
    +------------------+             +------------------+
              |
    父代理上下文保持干净。
    子代理上下文被丢弃。

核心洞察："全新的 messages=[] 实现了上下文隔离，父代理保持干净。"

注意：真正的 Claude Code 也使用进程内隔离（而非操作系统级别的进程 fork）。
子代理在同一进程中运行，拥有全新的消息数组和隔离的工具上下文 -- 与本教学实现相同的模式。

    与真正的 Claude Code 对比：
    +-------------------+------------------+----------------------------------+
    | 方面              | 本 demo          | 真正的 Claude Code               |
    +-------------------+------------------+----------------------------------+
    | 后端              | 仅进程内         | 5种后端: 进程内, tmux,            |
    |                   |                  | iTerm2, fork, remote             |
    | 上下文隔离        | 全新 messages=[] | createSubagentContext() 隔离     |
    |                   |                  | ~20个字段 (工具、权限、          |
    |                   |                  | 工作目录、环境变量、钩子等)      |
    | 工具过滤          | 手动筛选         | resolveAgentTools() 从父代理     |
    |                   |                  | 工具池中过滤; allowedTools       |
    |                   |                  | 替换所有允许规则                 |
    | 代理定义          | 硬编码系统提示词 | .claude/agents/*.md 带 YAML      |
    |                   |                  | frontmatter (AgentTemplate)      |
    +-------------------+------------------+----------------------------------+
"""

import os
import re
import sys
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from rich import print

load_dotenv(override=True)

# 将项目根目录加入 sys.path，以便导入 llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.router import call_llm_with_tools

WORKDIR = Path.cwd()

SYSTEM = f"""你是一个位于 {WORKDIR} 的编程代理。你是一个**指挥官**，不是执行者。

关于 `task` 工具的重要规则：
1. 你必须使用 `task` 工具来委派以下任何操作：
   - 读取或浏览文件和目录
   - 运行 shell 命令（ls、find、grep、git 等）
   - 分析代码结构或内容
   - 写入或编辑文件
   - 任何涉及文件系统交互的操作

2. 你绝对不能自己直接调用 bash、read_file、write_file 或 edit_file。
   必须始终通过 `task` 调用来委派给子代理执行。

3. 你的职责是：
   - 理解用户的请求
   - 将请求拆分为一个或多个子任务
   - 通过 `task` 工具委派每个子任务
   - 将子代理返回的结果整合为连贯的回复

4. 简单请求用一个 task 调用，复杂请求用多个 task 调用。

示例：如果用户问"这个项目有哪些文件？"，你应该调用：
  task(prompt="递归列出当前目录下的所有文件，并总结项目结构。", description="探索项目结构")

绝对不要直接执行工具。始终通过 `task` 委派。
"""

SUBAGENT_SYSTEM = f"""你是一个位于 {WORKDIR} 的编程子代理。你可以使用 bash、read_file、write_file 和 edit_file 工具。
请使用这些工具彻底完成给定的任务，然后提供一份清晰的发现和操作摘要。
摘要要简洁但完整。"""


class AgentTemplate:
    """
    从 Markdown frontmatter 解析代理定义。

    真正的 Claude Code 从 .claude/agents/*.md 加载代理定义。
    Frontmatter 字段：name, tools, disallowedTools, skills, hooks,
    model, effort, permissionMode, maxTurns, memory, isolation, color,
    background, initialPrompt, mcpServers。
    3种来源：内置的、自定义的(.claude/agents/)、插件提供的。
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


# -- 父代理和子代理共享的工具实现 --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    # 危险命令黑名单
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

# 子代理拥有所有基础工具，但没有 task 工具（防止递归生成子代理）-- OpenAI 格式
CHILD_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "运行 shell 命令。",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},

    {"type": "function", "function": {
        "name": "read_file", "description": "读取文件内容。",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},

    {"type": "function", "function": {
        "name": "write_file", "description": "将内容写入文件。",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},

    {"type": "function", "function": {
        "name": "edit_file", "description": "替换文件中的精确文本。",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]


def _call_with_system(messages, system_prompt, tools=None):
    """在 messages 前拼接 system 消息，然后调用统一的 call_llm_with_tools。"""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    return call_llm_with_tools(full_messages, tools=tools)


# -- 子代理：全新上下文、过滤后的工具、仅返回摘要 --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # 全新上下文
    for _ in range(30):  # 安全上限
        assistant_msg, finish_reason = _call_with_system(sub_messages, SUBAGENT_SYSTEM, CHILD_TOOLS)

        if assistant_msg is None:
            return "(子代理 API 调用失败)"

        # 将 assistant 回复加入子代理历史
        sub_messages.append({"role": "assistant", "content": assistant_msg.get("content"),
                             "tool_calls": assistant_msg.get("tool_calls")})

        # 如果模型没有调用工具，结束循环
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            break

        # 逐个执行工具调用，收集结果
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

    # 只有最终文本返回给父代理 -- 子代理上下文被丢弃
    last_assistant = sub_messages[-1] if sub_messages else {}
    if last_assistant.get("role") == "assistant" and last_assistant.get("content"):
        return last_assistant["content"]
    return "(无摘要)"


# -- 父代理工具：基础工具 + task 任务分发器 --
# 实际上这里就是：
# 父工具 = 子工具 + 一个 task 工具
PARENT_TOOLS = CHILD_TOOLS + [
    {"type": "function", "function": {
        "name": "task", "description": "生成一个拥有全新上下文的子代理。子代理与父代理共享文件系统，但不共享对话历史。",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "要委派给子代理的任务描述"}, "description": {"type": "string", "description": "任务的简短描述"}}, "required": ["prompt"]}}},
]


def agent_loop(messages: list):
    while True:
        assistant_msg, finish_reason = _call_with_system(messages, SYSTEM, PARENT_TOOLS)

        if assistant_msg is None:
            print("\033[31m[错误] API 调用失败\033[0m")
            return

        # 将 assistant 回复加入父代理历史
        messages.append({"role": "assistant", "content": assistant_msg.get("content"),
                         "tool_calls": assistant_msg.get("tool_calls")})

        # 如果模型没有调用工具，结束循环
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            return

        # 逐个执行工具调用，收集结果
        for tool_call in tool_calls:
            func = tool_call["function"]
            func_name = func["name"]
            try:
                args = json.loads(func["arguments"])
            except json.JSONDecodeError:
                args = {"command": func["arguments"]}

            if func_name == "task":
                desc = args.get("description", "子任务")
                prompt = args.get("prompt", "")
                print(f"\033[35m> 委派子任务 ({desc}): {prompt[:80]}\033[0m")
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

            # 打印最后一条 assistant 回复（父代理的最终整合结果）
        if history and history[-1].get("role") == "assistant":
            content = history[-1].get("content")
            if content:
                print(content)
        print()
