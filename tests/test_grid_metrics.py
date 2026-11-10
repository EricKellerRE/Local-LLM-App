import asyncio
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.grid_metrics import run_metrics, aggregate_export, render_report, discover_organizations
from local_model_app.plan_execution import PlanExecution
from tests.test_plan_execution import action
from tests.test_tool_coordinator import FakeManager, FakeModel, make_tool
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad


def mapping(case):
    return {'case_paths':[case],
        'branch':{'identity_fields':['id'],'voltage_fields':['kv1','kv2'],
            'attribution':[{'column':'owner','op':'eq','value':'Alpha'}],
            'length_field':'Miles','length_unit':'miles','rating_field':'MVA','rating_unit':'MVA'},
        'generator':{'identity_fields':['id'],'attribution':[{'column':'owner','op':'eq','value':'Alpha'}],
            'capacity_field':'MaxMW','capacity_unit':'MW'},'attribution_and_units_evidence':'Synthetic owner and unit headers'}


def export(path,text):
    path.write_text(text,encoding='utf-8')
    return {'structuredContent':{'ok':True,'artifact_complete':True,'artifact_path':str(path),
        'artifact_row_count':len(text.splitlines())-1,'artifact_skip_rows':0}}


class FixedGridTests(unittest.TestCase):

    def test_fixed_pipeline_owns_calls_paths_math_and_deduplication(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            branch=export(root/'opaque_branch.csv','id,Miles,MVA\na,10,1000\nb,20,2000\na,10,1000\n')
            gen=export(root/'opaque_gen.csv','id,MaxMW\na,700\nb,800\n')
            calls=[]; phases=[]
            async def call(args):
                calls.append(args)
                return branch if args['object_type']=='Branch' else gen
            result=asyncio.run(run_metrics(mapping('synthetic.pwb'),call,phases.append))['cases'][0]
            self.assertEqual([result[k] for k in ('miles_345kv','transmission_gw','generation_gw','gw_miles')],[30,3,1.5,90])
            self.assertEqual(result['branch_count'],2)
            self.assertEqual(len(calls),2)
            self.assertEqual(calls[0]['constraints'][-2:],[{'column':'kv1','op':'eq','value':345},{'column':'kv2','op':'eq','value':345}])
            self.assertEqual(phases,['branches','branch_totals','generators','calculation'])

    def test_incomplete_export_or_missing_measurement_is_not_zero(self):
        with TemporaryDirectory() as directory:
            result=export(Path(directory)/'x.csv','id,MW\na,\n')
            with self.assertRaisesRegex(ValueError,'nonnumeric'):
                aggregate_export(result,['id'],{'mw':'MW'})
            result['structuredContent']['artifact_complete']=False
            with self.assertRaisesRegex(ValueError,'complete'):
                aggregate_export(result,['id'],{'mw':'MW'})

    def test_empty_attribution_cannot_be_reported_as_zero_assets(self):
        with TemporaryDirectory() as directory:
            result=export(Path(directory)/'empty.csv','id,Miles,MVA\n')
            async def call(args): return result
            with self.assertRaisesRegex(ValueError,'Attribution is unresolved'):
                asyncio.run(run_metrics(mapping('synthetic.pwb'),call,lambda phase:None))

    def test_catalog_mw_overrides_model_gw(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            branch=export(root/'b.csv','id,Miles,MVA\na,30,3000\n')
            gen=export(root/'g.csv','id,GenMWMax\na,1500\n')
            m=mapping('synthetic.pwb'); m['generator'].update(capacity_field='GenMWMax',capacity_unit='GW')
            async def call(args): return branch if args['object_type']=='Branch' else gen
            result=asyncio.run(run_metrics(m,call,lambda phase:None,{'Gen':[{'variable_name':'GenMWMax','concise_name':'MWMax','description':'Maximum MW'}]}))
            self.assertEqual(result['cases'][0]['generation_gw'],1.5)
            self.assertEqual(result['cases'][0]['source_units']['capacity'],'MW')

    def test_organizational_discovery_reads_all_rows_and_tag_types(self):
        with TemporaryDirectory() as directory:
            result=export(Path(directory)/'org.csv','OwnerName,AreaName,ZoneName,OperatorName\nOther,Alpha Utilities,North,Operating Co\nOther,Alpha Utilities,South,Operating Co\n')
            fields=[{'variable_name':name} for name in ('OwnerName','AreaName','ZoneName','OperatorName')]
            calls=[]
            async def call(args): calls.append(args); return result
            inventories=asyncio.run(discover_organizations(['synthetic.pwb'],{'Branch':fields,'Gen':fields},call,lambda phase:None))
            self.assertEqual(inventories[0]['tags']['ZoneName'],['North','South'])
            self.assertEqual(inventories[0]['tags']['AreaName'],['Alpha Utilities'])
            self.assertEqual(len(calls),2)
            self.assertTrue(all(not c['constraints'] for c in calls))

    def test_export_headers_resolve_only_catalog_aliases(self):
        with TemporaryDirectory() as directory:
            result=export(Path(directory)/'x.csv','Asset ID,Max MW\na,1000\n')
            catalog=[{'variable_name':'ID','concise_name':'Asset ID'},
                     {'variable_name':'GenMWMax','concise_name':'Max MW'}]
            totals,count=aggregate_export(result,['ID'],{'capacity':'GenMWMax'},catalog)
            self.assertEqual(totals,{'capacity':1000})
            self.assertEqual(count,1)
            with self.assertRaisesRegex(ValueError,'exact exported headers'):
                aggregate_export(result,['ID'],{'capacity':'MadeUp'},catalog)
