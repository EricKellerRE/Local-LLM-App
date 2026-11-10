import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'Grid-Workshop'/'powerworld-aux-agent'))
from powerworld_aux_agent.tool_models import WorkspaceResolveInputsRequest
from powerworld_aux_agent.handlers import workspace

class DirectoryDiscoveryTests(unittest.TestCase):
    def test_directory_hint_searches_nested_cases_without_pww_or_default_roots(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory); nested=root/'nested';nested.mkdir()
            case=nested/'arbitrary-name.pwb';case.touch()
            with patch.object(workspace,'_default_case_search_roots',side_effect=AssertionError('Wrong scope')),patch.object(workspace,'_default_pww_search_roots',side_effect=AssertionError('Wrong scope')):
                result=workspace.workspace_resolve_inputs_tool(WorkspaceResolveInputsRequest(request_text='Find cases in this folder',case_path_hint=str(root)))
            self.assertTrue(result['ok'])
            self.assertTrue(result['case_found'])
            self.assertFalse(result['pww_found'])
            self.assertEqual(Path(result['resolved']['case_path']),case)
            self.assertEqual(result['case_candidate_count'],1)

    def test_multiple_cases_are_reported_without_returning_directory_as_file(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory)
            for name in ['first.pwb','second.pwb']:(root/name).touch()
            result=workspace.workspace_resolve_inputs_tool(WorkspaceResolveInputsRequest(request_text='Find all cases',case_path_hint=str(root)))
            self.assertTrue(result['ok'])
            self.assertIsNone(result['resolved']['case_path'])
            self.assertEqual(result['case_candidate_count'],2)
            self.assertEqual(len(result['alternatives']['case_candidates']),2)
