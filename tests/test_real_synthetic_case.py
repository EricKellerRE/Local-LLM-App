import unittest
from scripts.synthetic_grid_fixture import create_fixture, GRID
from src.session.case_session import CaseSession
from powerworld_aux_agent.object_query import QueryObjectsRequest, query_objects_tool

class RealSyntheticCaseTest(unittest.TestCase):
    def test_create_and_filter_synthetic_powerworld_case(self):
        case=create_fixture()
        request=QueryObjectsRequest(case_path=str(case), object_type='Branch',
            fields=['BusNum','BusNum:1','LineCircuit','LineLength','LineAMVA'],
            constraints=[{'column':'NomkVFrom','op':'eq','value':345},
                         {'column':'NomkVTo','op':'eq','value':345},
                         {'column':'AreaNum','op':'eq','value':1}],
            catalog_dir=str(GRID/'data/parsed/source_catalogs'),working_dir=str(case.parent))
        result=query_objects_tool(request)
        print('SYNTHETIC_CASE_PATH='+str(case),flush=True)
        self.assertEqual(result['matching_row_count'],2,result)
        self.assertTrue(result['artifact_complete'])
        catalog=CaseSession.load(GRID/'data/parsed/source_catalogs').catalog
        def total(result,object_type,canonical):
            spec=catalog.object_type(object_type).field_map()[canonical]
            header=next(name for name in [spec.variable_name,spec.concise_name] if name in result['columns'])
            return sum(float(row[header]) for row in result['rows'])
        self.assertEqual(total(result,'Branch','LineLength'),30)
        self.assertEqual(total(result,'Branch','LineAMVA'),3000)
        generation=query_objects_tool(QueryObjectsRequest(case_path=str(case),object_type='Gen',
            fields=['BusNum','GenID','GenMWMax'],constraints=[{'column':'AreaNum','op':'eq','value':1}],
            catalog_dir=str(GRID/'data/parsed/source_catalogs'),working_dir=str(case.parent)))
        self.assertEqual(generation['matching_row_count'],2)
        self.assertEqual(total(generation,'Gen','GenMWMax'),1500)
