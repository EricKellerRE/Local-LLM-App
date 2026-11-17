"""Host-owned scope, typed plan editing, and bounded observation memory."""
import copy
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath

from .analysis_plan import SCHEMA, obj, validate_plan
from .grid_metrics import catalog_unit, organizational_fields


def contract_for(state, cases):
    request=state['objective']
    digest=hashlib.sha256(request.encode()).hexdigest()
    contract=state.get('analysis_contract')
    if not contract or contract.get('request_digest')!=digest:
        contract={'version':1,'request_digest':digest,'request':request,
                  'case_paths':[], 'read_only':True,
                  'requires_objects':bool(re.search(r'\b(?:list|save|return|show)\b[^.\n]{0,100}\b(?:objects|lines|branches)\b|\bselection artifact',request,re.I))}
        state['analysis_contract']=contract
    if cases and not contract['case_paths']:
        roots=[PureWindowsPath(p) for p in re.findall(r'"([A-Za-z]:[\\/][^"\n]+)"',request)
               if not PureWindowsPath(p).suffix]
        explicit=[PureWindowsPath(p) for p in re.findall(r'"([A-Za-z]:[\\/][^"\n]+\.pwb)"',request,re.I)]
        if roots or explicit:
            if any(not (PureWindowsPath(case) in explicit or any(PureWindowsPath(case).is_relative_to(root) for root in roots)) for case in cases):
                raise ValueError('Discovered cases are outside the explicit requested scope; example repositories cannot substitute for that directory.')
        contract['case_paths']=list(dict.fromkeys(cases))
    if contract['case_paths'] and set(cases)!=set(contract['case_paths']):
        raise ValueError('Case scope changed after discovery. Restore the saved scope before execution.')
    return contract


def check_contract(contract, program, catalogs, execution=None):
    types=validate_plan(program,catalogs)
    if not contract['case_paths']: raise ValueError('No case scope has been resolved.')
    if contract['requires_objects'] and not any(types[o['source']]['kind']=='selection' for o in program['outputs']):
        raise ValueError('The request requires matching objects: include a selection in outputs, not only a count.')
    if execution is not None:
        if set(execution.get('cases',{}))!=set(contract['case_paths']):
            raise ValueError('The report does not cover every case in the saved scope.')
        if any(r['status'] not in ('completed','failed') for r in execution['cases'].values()):
            raise ValueError('A case is still unfinished.')


def draft_types(steps,catalogs):
    if not steps: return {}
    return validate_plan({'steps':steps,'outputs':[{'label':'step','source':s['id']} for s in steps]},catalogs)


def step_schema(steps,catalogs,field_hints=None):
    """Only offer operations with existing, correctly typed input handles."""
    types=draft_types(steps,catalogs)
    variants=[]
    for original in SCHEMA['properties']['steps']['items']['oneOf']:
        op=original['properties']['op']['const']
        if op=='select':
            for kind,fields in catalogs.items():
                variant=copy.deepcopy(original); props=variant['properties']
                props['object_type']={'const':kind}
                # Expand this bounded vocabulary through inspect_fields rather
                # than placing the full simulator catalog in the prompt.
                names=list(field_hints.get(kind,{})) if field_hints is not None else [f['variable_name'] for f in fields][:36]
                if not names: continue
                props['fields']['items']={'enum':names}
                props['filters']['items']['properties']['field']={'enum':names}
                org=[f['variable_name'] for f in organizational_fields(fields) if f['variable_name'] in names]
                if org: props['match']['properties']['fields']['items']={'enum':org}
                else: props.pop('match',None)
                variants.append(variant)
        elif op in ('filter','aggregate'):
            for handle,kind in types.items():
                if kind['kind']!='selection': continue
                variant=copy.deepcopy(original); props=variant['properties']
                props['source']={'const':handle}
                if op=='filter': props['filters']['items']['properties']['field']={'enum':sorted(kind['fields'])}
                else:
                    count=copy.deepcopy(variant)
                    count['properties']={key:value for key,value in count['properties'].items() if key in ('id','op','source','method')}
                    count['properties']['method']={'const':'count'}
                    variants.append(count)
                    known=[f for f in sorted(kind['fields']) if catalog_unit(catalogs[kind['object_type']],f)]
                    unknown=sorted(set(kind['fields'])-set(known))
                    for fields,established in ((known,True),(unknown,False)):
                        if not fields: continue
                        measure=copy.deepcopy(variant)
                        measure['properties']['field']={'enum':fields}
                        measure['properties']['method']={'enum':['sum','mean','min','max']}
                        measure['required'].append('field')
                        if established:
                            # Source units are evidence, not a model-selectable
                            # override. Conversion is a separate typed operation.
                            measure['properties'].pop('assumed_unit',None)
                            measure['properties'].pop('assumption',None)
                        variants.append(measure)
                    continue
                variants.append(variant)
        else:
            handles=[name for name,kind in types.items() if kind['kind']=='quantity']
            if not handles: continue
            variant=copy.deepcopy(original)
            for key in ('source',) if op=='convert' else ('left','right'):
                variant['properties'][key]={'enum':handles}
            variants.append(variant)
    return obj({'step':{'oneOf':variants}})


def append_step(state,step,catalogs):
    steps=state.setdefault('analysis_draft',[])
    if len(steps)>=32: raise ValueError('Plan step limit reached.')
    candidate=steps+[step]
    draft_types(candidate,catalogs)
    state['analysis_draft']=candidate
    return {'saved_step':step['id'],'handles':handles(candidate,catalogs)}


def handles(steps,catalogs):
    return {key:{**value,**({'fields':sorted(value['fields'])} if 'fields' in value else {})}
            for key,value in draft_types(steps,catalogs).items()}


def action_key(name,args):
    return hashlib.sha256(json.dumps([name,args],sort_keys=True).encode()).hexdigest()[:24]


def remember(state,name,args,feedback):
    ledger=state.setdefault('analysis_observations',{})
    ledger[action_key(name,args)]={'tool':name,'arguments':args,'feedback':feedback}


def observation_context(state,limit=6000):
    """Keep a bounded recent window; all observations remain durable on disk."""
    selected=[]; used=0
    for key,value in reversed(list(state.get('analysis_observations',{}).items())):
        record={'handle':key,**value}; size=len(json.dumps(record))
        if used+size>limit: continue
        selected.append(record); used+=size
    return list(reversed(selected))


def inspect_selection(selection,fields=None,offset=0,limit=10):
    if selection.get('kind')!='selection': raise ValueError('Expected a selection handle.')
    if not 1<=limit<=20 or offset<0: raise ValueError('Use limit 1..20 and a nonnegative offset.')
    rows=json.loads(Path(selection['path']).read_text(encoding='utf-8'))
    fields=fields or (list(rows[0])[:12] if rows else [])
    if len(fields)>12: raise ValueError('Inspect at most 12 fields at a time.')
    if rows and not set(fields)<=set(rows[0]): raise ValueError('Unknown selection fields.')
    page=[{f:str(row[f])[:200] for f in fields} for row in rows[offset:offset+limit]]
    return {'count':len(rows),'rows':page,'next_offset':offset+len(page) if offset+len(page)<len(rows) else None}


def partial_answer(state,reason):
    """Report supported findings on pause; never turn absent evidence into zero."""
    lines=['The analysis is incomplete. Here is what the saved evidence establishes so far.','']
    execution=state.get('analysis_execution',{})
    program=state.get('analysis_program') or state.get('analysis_proposed_program')
    if program and execution.get('cases'):
        from .analysis_plan import render
        lines.extend([render(program,execution),''])
    contract=state.get('analysis_contract',{})
    cases=contract.get('case_paths',[])
    completed=sum(r.get('status')=='completed' for r in execution.get('cases',{}).values())
    if cases: lines.append(f'Cases discovered: {len(cases)}. Cases with completed analysis: {completed}.')
    findings=[]
    for observation in state.get('analysis_observations',{}).values():
        if observation['tool']!='host__inspect_tags': continue
        args=observation['arguments']; feedback=observation['feedback']
        index=args['case_index']; case=cases[index-1] if index<=len(cases) else 'case '+str(index)
        text=args['text'].replace('\n',' ')[:160]
        label=case.replace('\\','/').rsplit('/',1)[-1]+' — '+args['object_type']+' organizational search for “'+text+'”'
        matches=feedback.get('matches',[])
        if matches:
            details='; '.join(str(m['field'])+' = '+str(m['value'])[:200]+' ('+str(m['count'])+' objects)' for m in matches[:8])
            findings.append(label+': '+details+'.'+(' Additional matches remain in the saved observation.' if len(matches)>8 or feedback.get('truncated') else ''))
        else:
            findings.append(label+': no matching tags on this inspected page. This does not establish that the entity has no managed assets.')
    if findings:
        lines.extend(['']+['- '+f for f in findings[-12:]])
        lines.extend(['','These are organizational-tag observations, not a completed managed-line selection. Area/zone membership alone does not establish ownership or operation. Results cannot be generalized to unprocessed cases.'])
    elif not execution.get('cases'):
        lines.extend(['','No asset-level findings have been validated yet; I cannot conclude either that matching lines exist or that there are none.'])
    lines.extend(['','Unresolved: the requested analysis and case coverage are not complete.','Reason: '+reason,'Saved evidence and completed work remain available for resume.'])
    return '\n'.join(lines)
