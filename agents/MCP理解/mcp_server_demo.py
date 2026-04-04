#!/usr/bin/env python3
"""
mcp_server_demo.py - 最简单的 MCP Server 示例

这个文件就是"工具提供方"，把工具写成 MCP 格式。
任何支持 MCP 的 AI Agent 都能自动发现并调用这些工具。

运行方式：不需要手动运行，Client 会自动启动它。
"""

from mcp.server.fastmcp import FastMCP
from datetime import datetime


# 创建一个 MCP Server 实例，名字叫 "demo-tools"
mcp = FastMCP("demo-tools")


# ========== 工具1：加法 ==========
# 只需要加一个 @mcp.tool() 装饰器，SDK 会自动：
#   1. 从函数签名 (a: int, b: int) 自动生成 JSON Schema
#   2. 从 docstring 自动生成工具描述
#   3. 自动注册路由，Client 调用时自动找到这个函数
@mcp.tool()
def add(a: int, b: int) -> str:
    """两个数字相加。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return str(a + b)


# ========== 工具2：获取当前时间 ==========
@mcp.tool()
def get_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========== 启动 Server ==========
# 通过 stdio（标准输入输出）通信，Client 会启动这个进程并通过管道和它对话
if __name__ == "__main__":
    mcp.run(transport="stdio")
