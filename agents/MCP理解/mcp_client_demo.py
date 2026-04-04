#!/usr/bin/env python3
"""
mcp_client_demo.py - 最简单的 MCP Client 示例

这个文件就是"Agent 使用方"，它会：
1. 启动 MCP Server 进程
2. 自动发现 Server 提供了哪些工具（tools/list）
3. 调用这些工具（tools/call）
4. 拿到结果

运行方式：python agents/mcp_client_demo.py
"""

import asyncio
import sys
import os

# 修复 Windows 终端编码问题
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    # ========== 第1步：告诉 Client 怎么启动 Server ==========
    # 用 __file__ 算出 Server 的绝对路径，这样无论从哪个目录运行都不会出错
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "mcp_server_demo.py")

    server_params = StdioServerParameters(
        command=sys.executable,  # 用当前 Python 解释器
        args=[server_script],  # 启动我们的 MCP Server（绝对路径）
    )

    # ========== 第2步：连接到 Server ==========
    print("=" * 50)
    print("[*] 正在连接 MCP Server...")
    print("=" * 50)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            print("[OK] 连接成功!\n")

            # ========== 第3步：自动发现工具（tools/list）==========
            # 这就是 MCP 的核心！Client 不需要提前知道有哪些工具
            # 它会问 Server："你有哪些工具？"
            tools_result = await session.list_tools()

            print("[Tools] 发现的工具列表：")
            print("-" * 40)
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    参数: {tool.inputSchema}")
            print()

            # ========== 第4步：调用工具（tools/call）==========
            # 调用 add 工具
            print("[Call] 调用 add(3, 5)...")
            result = await session.call_tool("add", {"a": 3, "b": 5})
            print(f"  结果: {result.content[0].text}")
            print()

            # 调用 get_time 工具
            print("[Call] 调用 get_time()...")
            result = await session.call_tool("get_time", {})
            print(f"  结果: {result.content[0].text}")
            print()

            print("=" * 50)
            print("[OK] 演示完成!")
            print("=" * 50)
            print()
            print("[Key Points]")
            print("  1. Client 不需要提前知道 Server 有哪些工具")
            print("  2. Client 通过 list_tools() 自动发现工具")
            print("  3. Client 通过 call_tool() 调用工具")
            print("  4. Server 和 Client 通过标准协议通信，完全解耦")
            print("  5. 同一个 Server 可以被任何 MCP Client 使用!")


if __name__ == "__main__":
    asyncio.run(main())
