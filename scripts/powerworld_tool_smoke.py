from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a non-simulator PowerWorld tool-loop smoke test.")
    parser.add_argument("--powerworld-repo", type=Path, required=True)
    parser.add_argument("--model-url", default="http://127.0.0.1:8765")
    parser.add_argument("--tool-url", default="http://127.0.0.1:8000")
    parser.add_argument("--transcript", type=Path, default=Path("data/powerworld-tool-smoke.json"))
    return parser.parse_args()


class BridgeExecutor:
    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self.request_id = 0

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.request_id += 1
        response = self.bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if not response or "result" not in response:
            raise RuntimeError(f"PowerWorld bridge returned an invalid response: {response!r}")
        result = response["result"]
        return result.get("structuredContent", result)


def main() -> int:
    args = parse_args()
    repo = args.powerworld_repo.resolve()
    sys.path.insert(0, str(repo))

    from powerworld_aux_agent.agent_loop import AgentSession, LmStudioChatCompletionsClient
    from powerworld_aux_agent.mcp_server import META_TOOLS, PowerWorldMcpBridge

    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in META_TOOLS
    ]
    system_prompt = (
        "You are running a safe integration test of the staged PowerWorld tool surface. "
        "First call powerworld_route_request with the user's complete request. "
        "After it returns, call powerworld_list_tools using exactly the capability it selected. "
        "After that result, stop calling tools and briefly report the selected capability and the listed backend tool names. "
        "Never call powerworld_call_tool or any direct ranking tool during this test."
    )
    session = AgentSession(
        model_client=LmStudioChatCompletionsClient(
            base_url=args.model_url,
            api_token="",
            model="gemma-4-12b-it",
            temperature=0.0,
            max_tokens=192,
            timeout=900,
        ),
        tool_executor=BridgeExecutor(PowerWorldMcpBridge(args.tool_url)),
        tools=tools,
        max_steps=4,
        transcript_path=args.transcript.resolve(),
        system_prompt=system_prompt,
    )
    answer = session.run_turn(
        "Classify and disclose the tools for this request: compare two PowerWorld planning cases. "
        "This is discovery only; do not open a case or run the simulator."
    )
    print(json.dumps({"answer": answer, "transcript": str(args.transcript.resolve())}, indent=2))
    return 0 if not answer.startswith("[error]") else 1


if __name__ == "__main__":
    raise SystemExit(main())
