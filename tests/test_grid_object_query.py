"""Synthetic-only contract tests for the Grid object-query tool."""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'Grid-Workshop' / 'powerworld-aux-agent'))
from powerworld_aux_agent import object_query as query_module
Request = query_module.QueryObjectsRequest


class ObjectQueryTests(unittest.TestCase):
    def test_filter_field_is_included_and_catalog_alias_resolved(self):
        spec = SimpleNamespace(variable_name='VoltageCanonical', concise_name='kV')
        fields = {'VoltageCanonical':spec, 'kV':spec}
        request = Request(case_path='synthetic.pwb',object_type='Branch',fields=['ID'],
                          constraints=[{'column':'VoltageCanonical','op':'eq','value':345}])
        self.assertEqual(query_module.constraint_fields(request,fields),['ID','VoltageCanonical'])
        resolved = query_module.exported_constraints(request,fields,['ID','kV'])
        self.assertEqual(resolved.constraints[0]['column'],'kV')
        self.assertEqual(request.constraints[0]['column'],'VoltageCanonical')
        self.assertEqual(request.fields,['ID'])
        source = SimpleNamespace(fields=['ID','kV'],rows=[{'ID':'one','kV':'345'},{'ID':'two','kV':'138'}],export_path='synthetic.csv')
        with TemporaryDirectory(dir=ROOT) as directory:
            result = query_module.filter_export(source,resolved,Path(directory)/'filtered.csv')
        self.assertEqual(result['matching_row_count'],1)
        self.assertEqual(result['rows'][0]['ID'],'one')

    def test_unknown_filter_field_is_not_guessed(self):
        request = self.query(constraints=[{'column':'unknown','op':'eq','value':345}])
        self.assertEqual(query_module.constraint_fields(request,{}), request.fields)
        self.assertEqual(query_module.exported_constraints(request,{},['kV']).constraints,request.constraints)

    def query(self, **kwargs):
        return Request(case_path='synthetic.pwb', object_type='Branch', fields=['ID','kV','Entity'], **kwargs)

    def test_complete_export_despite_paged_response_and_numeric_membership(self):
        rows = [{'ID':str(i),'kV':'345.00','Entity':'Alpha'} for i in range(250)]
        rows += [{'ID':'other','kV':'138','Entity':'Alpha'}, {'ID':'other2','kV':'345','Entity':'Beta'}]
        source = SimpleNamespace(fields=['ID','kV','Entity'],rows=rows,export_path='synthetic-source.csv')
        request = self.query(constraints=[{'column':'kV','op':'in','value':[345]}, {'column':'Entity','op':'eq','value':'alpha','case_sensitive':False}],limit=100)
        with TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory)/'matches.csv'
            result = query_module.filter_export(source,request,path)
            with path.open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),250)
            self.assertEqual(result['source_row_count'],252)
            self.assertEqual(result['matching_row_count'],250)
            self.assertEqual(result['returned_row_count'],100)
            self.assertEqual(result['next_offset'],100)
            self.assertTrue(result['artifact_complete'])
            self.assertTrue(result['rows_truncated'])
            request.offset=200
            final = query_module.filter_export(source,request,path)
            self.assertEqual(final['returned_row_count'],50)
            self.assertIsNone(final['next_offset'])

    def test_unknown_column_rejected_even_for_empty_export(self):
        source=SimpleNamespace(fields=['ID'],rows=[],export_path='source.csv')
        with TemporaryDirectory(dir=ROOT) as directory:
            with self.assertRaisesRegex(ValueError,'Unknown constraint'):
                query_module.filter_export(source,self.query(constraints=[{'column':'wrong','op':'eq','value':0}]),Path(directory)/'output.csv')

    def test_zero_and_nonfinite_values(self):
        rule={'column':'x','op':'eq','value':0}
        self.assertTrue(query_module._matches('0.0',rule))
        for value in ['',None,'NaN','Infinity']:
            self.assertFalse(query_module._matches(value,rule))
        with self.assertRaises(ValueError):
            self.query(constraints=[{'column':'kV','op':'gt','value':'unknown'}])
        with self.assertRaises(ValueError):
            self.query(constraints=[{'column':'kV','op':'in','value':345}])

    def test_handler_fetches_full_inventory_and_closes_without_saving(self):
        session=MagicMock()
        session.catalog.object_type.return_value=SimpleNamespace(object_type='Branch',field_map=lambda:{})
        session.fetch_rows.return_value=SimpleNamespace(fields=['ID','kV','Entity'],rows=[],export_path='source.csv')
        with TemporaryDirectory(dir=ROOT) as directory:
            with patch.object(query_module.CaseSession,'load') as load, patch.object(query_module,'working_dir',return_value=Path(directory)), patch.object(query_module,'catalog_dir',return_value=Path(directory)):
                load.return_value.__enter__.return_value=session
                from powerworld_aux_agent.tool_registry import invoke_tool
                result=invoke_tool('case.query_objects', self.query().__dict__)
            self.assertEqual(result['matching_row_count'],0)
            self.assertEqual(result['rows'],[])
            self.assertTrue(result['artifact_complete'])
        session.close_case.assert_called_once()
        session.save_case.assert_not_called()
        args,kwargs=session.fetch_rows.call_args
        self.assertEqual(args,('Branch',['ID','kV','Entity']))
        self.assertEqual(set(kwargs),{'filename'})


if __name__=='__main__':
    unittest.main()
