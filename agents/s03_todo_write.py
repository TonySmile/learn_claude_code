#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
"""

import os
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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[待办]", "in_progress": "[进行中]", "completed": "[已完成]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- Tool implementations --
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
        # Windows 下 cmd 输出为 GBK 编码，优先尝试 GBK 解码，失败则回退 UTF-8
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, timeout=120)
        raw = r.stdout + r.stderr
        try:
            out = raw.decode("gbk").strip()
        except (UnicodeDecodeError, AttributeError):
            out = raw.decode("utf-8", errors="replace").strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
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
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

# OpenAI 格式的工具定义
TOOLS = [
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

    {"type": "function", "function": {
        "name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
        "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}}},
]


# -- Agent loop with nag reminder injection --
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # 构建带 system 消息的完整消息列表
        full_messages = [{"role": "system", "content": SYSTEM}] + messages

        # 调用 Venus API
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
        used_todo = False
        for tool_call in tool_calls:
            func = tool_call["function"]
            func_name = func["name"]
            try:
                args = json.loads(func["arguments"])
            except json.JSONDecodeError:
                args = {"command": func["arguments"]}

            handler = TOOL_HANDLERS.get(func_name)
            try:
                output = handler(**args) if handler else f"Unknown tool: {func_name}"
            except Exception as e:
                output = f"Error: {e}"

            print(f"\033[33m> {func_name}: {str(output)[:1000]}\033[0m")

            if func_name == "todo":
                used_todo = True

            # OpenAI 格式：tool 角色的消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(output),
            })

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

        # Nag reminder: 连续3轮没更新 todo，注入提醒
        if rounds_since_todo >= 3:
            messages.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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

        # 帮我看看我们现在在什么目录，然后帮我总结一下GH_try.py这个文件的内容。必须要使用todo 这个工具

