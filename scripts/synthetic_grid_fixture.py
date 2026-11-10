"""Create only a fresh, explicitly synthetic case in the test workspace."""
import sys, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GRID=ROOT.parent/'Grid-Workshop'/'powerworld-aux-agent'
sys.path.insert(0,str(GRID))
from src.session.case_session import CaseSession
from src.executor.simauto_client import SimAutoClient, _simauto_error_text
from src.compiler.object_list_renderer import render_data_section

def create_fixture(scale=1):
    import pythoncom
    import win32com.client
    folder=ROOT/'synthetic-workflow-check'/('case-'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    case=folder/'synthetic-alpha.pwb'
    client=SimAutoClient()
    pythoncom.CoInitialize()
    client._app=win32com.client.DispatchEx(client.prog_id)
    client.connected=True
    def checked(result):
        error=_simauto_error_text(result)
        if error: raise RuntimeError(error)
    try:
        with CaseSession.load(GRID/'data/parsed/source_catalogs',simauto_client=client,working_dir=folder) as session:
            checked(session.new_case())
            checked(session.enter_mode('EDIT'))
            sections=[
                ('Area',['AreaNum','AreaName'],[[1,'Alpha'],[2,'Beta']]),
                ('Zone',['ZoneNum','ZoneName'],[[1,'Synthetic']]),
                ('Bus',['BusNum','BusName','BusNomVolt','AreaNum','ZoneNum'],
                 [[i,'Synthetic'+str(i),345 if i<6 else 138,2 if i in (4,5) else 1,1] for i in range(1,8)]),
                ('Branch',['BusNum','BusNum:1','LineCircuit','LineR','LineX','LineLength','LineAMVA','LineBMVA','LineCMVA','LineStatus'],
                 [[1,2,'1',.01,.1,10,1000,1000,1000,'Closed'],[2,3,'1',.01,.1,20,2000,2000,2000,'Closed'],
                  [4,5,'1',.01,.1,70,4000,4000,4000,'Closed'],[6,7,'1',.01,.1,99,9000,9000,9000,'Closed']]),
                ('Gen',['BusNum','GenID','GenMWMax','GenMW','GenStatus','GenMVR','GenAGCAble','GenMWMin','GenAVRAble','GenVoltSet','GenMVRMin','GenMVRMax'],
                 [[bus,'1',mw,0,'Closed',0,'NO',0,'YES',1,-100,100] for bus,mw in [(1,700),(2,800),(4,9000)]]),
            ]
            for object_type,fields,rows in sections:
                for field in ('LineLength', 'LineAMVA', 'LineBMVA', 'LineCMVA', 'GenMWMax'):
                    if field in fields:
                        index = fields.index(field)
                        for row in rows:
                            row[index] *= scale
                result=session.process_aux_text(render_data_section(object_type,fields,rows),filename=object_type+'.aux')
                if not result.success:
                    raise RuntimeError(f'Synthetic {object_type} creation failed: {result.stderr}')
                observed=session.fetch_rows(object_type,fields,filename=object_type+'-verify.csv')
                if len(observed.rows) != len(rows):
                    raise RuntimeError(f'Synthetic {object_type} fixture has {len(observed.rows)} rows; expected {len(rows)}.')
            checked(session.save_case(case))
            checked(session.close_case())
        return case
    finally:
        client.close()
