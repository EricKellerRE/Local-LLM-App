"""Opaque object selections: company search, then deterministic calculation."""
import csv
import json
import re
import uuid
from pathlib import Path

from .atomic_files import write_json
from .grid_metrics import organizational_fields, unpack, aggregate_export
from .quantity_units import convert
from .cancellation import check_stopped


CORE={'Branch':['BusNum','BusNum:1','LineCircuit','BusNomVolt','BusNomVolt:1','LineLength','LineAMVA'],
      'Gen':['BusNum','GenID','GenMWMax']}
IDENTITY={'Branch':['BusNum','BusNum:1','LineCircuit'],'Gen':['BusNum','GenID']}


def matches_company(value, company):
    # Whole phrase, case/punctuation insensitive. No invented acronym expansion.
    normalize=lambda s:' '.join(re.findall(r'[\w]+',str(s).casefold()))
    actual,wanted=normalize(value),normalize(company)
    return bool(wanted) and (' '+wanted+' ') in (' '+actual+' ')


async def select_objects(cases, company, catalogs, call, directory, checkpoint):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest={'selection_id':uuid.uuid4().hex,'company':company,'cases':[]}
    for index,case in enumerate(cases):
        check_stopped(); item={'case':case,'objects':{}}; manifest['cases'].append(item)
        for kind in ('Branch','Gen'):
            checkpoint('search_'+kind.lower(),index+1,len(cases))
            try:
                catalog=catalogs[kind]; names={f['variable_name'] for f in catalog}
                missing=set(CORE[kind])-names
                if missing: raise ValueError('Required catalog fields unavailable: '+', '.join(sorted(missing)))
                org=organizational_fields(catalog)
                if not org: raise ValueError('No organizational fields are available.')
                fields=list(dict.fromkeys(CORE[kind]+[f['variable_name'] for f in org]))
                data=unpack(await call({'case_path':case,'object_type':kind,'fields':fields,'constraints':[],'limit':1}))
                if not data.get('artifact_complete'): raise ValueError('Search needs a complete object export.')
                destination=directory/(manifest['selection_id']+'-'+str(index)+'-'+kind+'.csv')
                selected=0; source_count=0; evidence=set()
                with Path(data['artifact_path']).open(encoding='utf-8-sig',newline='') as stream, destination.open('w',encoding='utf-8',newline='') as output:
                    for _ in range(int(data.get('artifact_skip_rows',0))): next(stream)
                    reader=csv.DictReader(stream)
                    aliases={f['variable_name']:f.get('concise_name') for f in catalog}
                    headers={name:next((h for h in (name,aliases.get(name)) if h in (reader.fieldnames or [])),None) for name in fields}
                    if any(not headers[n] for n in CORE[kind]): raise ValueError('Required measurement or identity columns missing from export.')
                    # Do not claim an exhaustive organizational search if an exported tag is absent.
                    if any(not headers[f['variable_name']] for f in org): raise ValueError('Some organizational fields were not exported; search is incomplete.')
                    writer=csv.DictWriter(output,fieldnames=CORE[kind]); writer.writeheader()
                    for row in reader:
                        source_count+=1
                        hits=[(f['variable_name'],row.get(headers[f['variable_name']],'')) for f in org
                              if matches_company(row.get(headers[f['variable_name']],''),company)]
                        if hits:
                            selected+=1; evidence.update(hits)
                            writer.writerow({name:row[headers[name]] for name in CORE[kind]})
                if source_count!=data.get('artifact_row_count'): raise ValueError('Object export row count mismatch.')
                item['objects'][kind]={'path':str(destination),'count':selected,'source_count':source_count,
                                       'evidence':[{'field':f,'value':v} for f,v in sorted(evidence)]}
            except PermissionError: raise
            except (ValueError,KeyError,TypeError,OSError) as exc:
                item['objects'][kind]={'error':str(exc)}
            write_json(directory/(manifest['selection_id']+'.json'),manifest)
    return manifest


def calculate_selection(manifest):
    reports=[]
    for item in manifest['cases']:
        check_stopped()
        row={'case':item['case'],'miles':None,'transmission':None,'generation':None,'gw_miles':None,'notes':[]}
        for kind in ('Branch','Gen'):
            obj=item['objects'].get(kind,{})
            if obj.get('error'):
                row['notes'].append(kind+': '+obj['error']); continue
            if not obj.get('count'):
                row['notes'].append(kind+': no organizational tag matched; attribution unresolved.'); continue
            try:
                path=Path(obj['path'])
                # Filter only the saved objects; no backend request or model transcription.
                if kind=='Branch':
                    with path.open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
                    chosen=[r for r in rows if float(r['BusNomVolt'])==345 and float(r['BusNomVolt:1'])==345]
                    path=path.with_suffix('.345.csv')
                    with path.open('w',encoding='utf-8',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=CORE[kind]); writer.writeheader(); writer.writerows(chosen)
                    count=len(chosen); metrics={'length':'LineLength','rating':'LineAMVA'}
                else: count=obj['count']; metrics={'capacity':'GenMWMax'}
                sums,unique=aggregate_export({'artifact_complete':True,'artifact_path':str(path),'artifact_row_count':count},IDENTITY[kind],metrics)
                if kind=='Branch':
                    row['miles']=sums['length']; row['transmission']=convert(sums['rating'],'MVA','GW',1)['value']
                    row['gw_miles']=row['miles']*row['transmission']
                else: row['generation']=convert(sums['capacity'],'MW','GW')['value']
                row['notes'].append(kind+': '+str(unique)+' matching objects; '+ '; '.join(e['field']+' = '+e['value'] for e in obj['evidence']))
            except (ValueError,KeyError,TypeError,OSError) as exc: row['notes'].append(kind+': '+str(exc))
        reports.append(row)
    return reports


def render_selection(manifest,reports):
    def safe(text): return str(text).replace('|','\\|').replace('\n',' ')
    def number(value): return 'Unknown' if value is None else f'{value:,.6f}'.rstrip('0').rstrip('.')
    lines=['Results for **'+safe(manifest['company'])+'**, searching all available owner, operator, company, area and zone tags in each discovered case.','',
           '| Case | 345 kV miles | Transmission GW | Generation GW | GW-miles |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row in reports:
        lines.append('| '+safe(row['case'].replace('\\','/').rsplit('/',1)[-1])+' | '+' | '.join(number(row[k]) for k in ('miles','transmission','generation','gw_miles'))+' |')
    lines+=['','Both branch endpoints must be 345 kV. LineAMVA is converted from MVA with PF=1; GenMWMax is converted from MW. LineLength is assumed to be miles because its catalog does not specify a unit. GW-miles is total miles × transmission GW. Organizational-tag matches are the attribution basis; an area or zone match alone does not prove ownership or operation.','']
    for row in reports:
        lines.append('**'+safe(row['case'].replace('\\','/').rsplit('/',1)[-1])+'**')
        lines.extend('- '+safe(note) for note in row['notes']); lines.append('')
    complete=sum(all(row[k] is not None for k in ('miles','transmission','generation','gw_miles')) for row in reports)
    lines.append(f'Cases discovered: {len(manifest["cases"])}. Completed: {complete}. Unresolved or failed: {len(reports)-complete}.')
    return '\n'.join(lines)
