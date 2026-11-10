"""Run the real local GGUF planner against a case-free Grid Workshop tool."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_model_app.config import Settings
from local_model_app.gguf_model import GgufModel
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad


async def main():
    settings = replace(Settings.from_environment(ROOT), model_id=sys.argv[1], model_kind='gguf',
                       device='cpu', context_window=16384, reasoning_budget=0, max_new_tokens=512,
                       planner_max_new_tokens=128, tool_action_max_new_tokens=256)
    model = GgufModel(settings, telemetry_path=ROOT/'data/gguf-grid-smoke/telemetry.jsonl')
    manager = McpPluginManager(McpPluginRegistry(ROOT/'config/mcp.d', project_root=ROOT))
    activity = ROOT/'data/gguf-grid-smoke/activity.jsonl'
    start_size = activity.stat().st_size if activity.exists() else 0
    try:
        coordinator = ToolCoordinator(model, manager, Scratchpad(ROOT/'data/gguf-grid-smoke/scratchpad.jsonl'), activity)
        answer = await coordinator.respond('Use regulatory.list_tests to list available regulatory study profiles. Do not open any case or run a study. Report the number returned.', [], ['grid-workshop.powerworld'])
        with activity.open('rb') as handle:
            handle.seek(start_size)
            events = [json.loads(line) for line in handle]
        calls = [event['tool'] for event in events if event['event']=='tool_call']
        assert 'grid__regulatory_list_tests' in calls, calls
        assert any(event['event']=='tool_result' and not event['result'].get('isError') for event in events)
        print(json.dumps({'answer':answer,'tool_calls':calls,'case_files_opened':0}, indent=2))
    finally:
        await manager.stop_all()
        model.close()


if __name__ == '__main__':
    asyncio.run(main())
