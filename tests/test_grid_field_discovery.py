import unittest
from tests.test_grid_object_query import ROOT
from src.session.case_session import CaseSession

class FieldDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=CaseSession.load(ROOT.parent/'Grid-Workshop/powerworld-aux-agent/data/parsed/source_catalogs').catalog
    def test_generator_alias_resolves_to_canonical_type(self):
        self.assertEqual(self.catalog.object_type('Generator').object_type,'Gen')
        self.assertEqual(self.catalog.object_type('branch').object_type,'Branch')
    def test_multiconcept_search_preserves_every_concept(self):
        spec=self.catalog.object_type('Branch')
        concepts=['Line Length','MVA rating','Voltage','AreaName']
        expected={spec.search_fields(q,limit=1)[0].variable_name for q in concepts}
        actual={f.variable_name for f in spec.search_fields(', '.join(concepts),limit=4)}
        self.assertEqual(actual,expected)
        self.assertIn('LineLength',actual)
        self.assertIn('LineAMVA',actual)
