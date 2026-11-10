"""Exercise the installed Grid plugin using synthetic CSV data only."""
import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_model_app.config import _load_dotenv
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.mcp_manager import McpPluginManager


async def main():
    _load_dotenv(ROOT / '.env')
    manager = McpPluginManager(McpPluginRegistry(ROOT / 'config/mcp.d', project_root=ROOT))
    try:
        status = await manager.start('grid-workshop.powerworld')
        tools = manager.tools_for_plugins(['grid-workshop.powerworld'])
        required = {'workspace.resolve_inputs', 'case.search_object_fields', 'case.query_objects', 'session.fetch_rows', 'results.summarize_artifact'}
        assert required <= {t.native_name for t in tools}
        tool = next(t for t in tools if t.native_name == 'results.summarize_artifact')
        assert {'filters', 'skip_rows'} <= tool.input_schema['properties'].keys()
        with TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'synthetic.csv'
            from local_model_app.tool_coordinator import ToolCoordinator
            from local_model_app.scratchpad import Scratchpad
            from types import SimpleNamespace
            coordinator = ToolCoordinator(SimpleNamespace(), manager, Scratchpad(Path(directory)/'scratch.jsonl'), Path(directory)/'activity.jsonl')
            failure = await coordinator._execute(tool, {'path': str(Path(directory)/'nonexistent.csv')}, [])
            assert failure['isError'] and 'Backend detail:' in failure['message'], failure
            # Filename placeholders only: discovery must not open these as cases.
            placeholder = Path(directory)/'different-name.pwb'
            placeholder.touch()
            resolver = next(t for t in tools if t.native_name == 'workspace.resolve_inputs')
            resolved = await manager.call_tool(resolver.exposed_name, {'request_text':f'Find cases in {directory}', 'case_search_roots':[directory], 'pww_search_roots':[directory]})
            assert str(placeholder) in json.dumps(resolved).replace('\\\\', '\\'), resolved
            path.write_text('Synthetic metadata\nEntity,kV,Miles,Rating\nAlpha,345,10,1000\nAlpha,345,20,2000\nAlpha,138,99,9000\nBeta,345,5,500\nGamma,345,7,NaN\n', encoding='utf-8')
            async def summarize(**extra):
                result = await manager.call_tool(tool.exposed_name, {'path': str(path), 'skip_rows': 1, 'sample_rows': 0, **extra})
                assert not result.get('isError'), result
                return result['structuredContent']
            result = await summarize(filters=[{'column':'Entity','op':'eq','value':'Alpha'}, {'column':'kV','op':'eq','value':345}])
            assert result['source_row_count'] == 5 and result['row_count'] == 2
            assert result['numeric_columns']['Miles']['sum'] == 30
            assert result['numeric_columns']['Rating']['sum'] == 3000
            grouped = await summarize(group_by='Entity', top_n=1)
            assert grouped['group_summary']['groups_truncated'] is True
            missing = await summarize(filters=[{'column':'Entity','op':'eq','value':'Gamma'}])
            assert 'Rating' not in missing['numeric_columns']
            empty = await summarize(filters=[{'column':'Entity','op':'eq','value':'Absent'}])
            assert empty['row_count'] == 0
            from mcp.shared.exceptions import MCPError
            try:
                bad = await manager.call_tool(tool.exposed_name, {'path':str(path), 'skip_rows':1, 'metric':'wrong_column'})
            except MCPError as exc:
                assert '400' in str(exc)
            else:
                assert bad.get('isError'), 'Unknown columns must fail explicitly'
        print(json.dumps({'ok':True, 'tools':status['tool_count'], 'checks':['required tools discovered', 'backend error detail preserved', 'directory input discovery', 'full export filtering', 'metadata header', 'group truncation', 'nonfinite values', 'empty selection', 'unknown column rejection'], 'case_files_opened':0}, indent=2))
    finally:
        print('Readiness checks: shutting down plugin.',flush=True)
        await manager.stop_all()
        print('Readiness checks: shutdown complete.',flush=True)


if __name__ == '__main__':
    asyncio.run(main())
