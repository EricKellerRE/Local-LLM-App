"""Validated, reusable per-case analysis operations. No model-generated Python."""
import csv
import hashlib
import json
import math
from pathlib import Path

from jsonschema import validate
from .atomic_files import write_json
from .cancellation import check_stopped
from .grid_metrics import catalog_unit, organizational_fields, unpack
from .grid_selection import matches_company
from .quantity_units import convert

IDENTITIES = {'Branch':['BusNum','BusNum:1','LineCircuit'], 'Gen':['BusNum','GenID'], 'Bus':['BusNum']}

def obj(properties, required=None):
    return {'type':'object','properties':properties,'required':list(properties) if required is None else required,'additionalProperties':False}

STRING={'type':'string','minLength':1}
ID={'type':'string','pattern':'^[A-Za-z][A-Za-z0-9_]{0,39}$'}
FILTER=obj({'field':STRING,'op':{'enum':['eq','ne','in','lt','le','gt','ge','contains']},
            'value':{'type':['string','number','boolean','array']}})
FILTERS={'type':'array','items':FILTER,'maxItems':20}
def operation(name, props, required=None):
    return obj({'id':ID,'op':{'const':name},**props},['id','op',*(list(props) if required is None else required)])

SCHEMA=obj({
    'steps':{'type':'array','minItems':1,'maxItems':32,'items':{'oneOf':[
        operation('select',{'object_type':{'enum':list(IDENTITIES)},'fields':{'type':'array','items':STRING,'uniqueItems':True},
            'filters':FILTERS,'match':obj({'text':STRING,'fields':{'type':'array','items':STRING}},['text'])},['object_type','fields']),
        operation('filter',{'source':ID,'filters':FILTERS}),
        operation('aggregate',{'source':ID,'method':{'enum':['count','sum','mean','min','max']},'field':STRING,
            'assumed_unit':{'enum':['MW','GW','MVA','GVA','miles','km','kV','unknown']},'assumption':STRING},['source','method']),
        operation('convert',{'source':ID,'unit':{'enum':['MW','GW','MVA','GVA','miles','km']},'power_factor':{'type':'number','minimum':0,'maximum':1}},['source','unit']),
        operation('calculate',{'left':ID,'right':ID,'method':{'enum':['add','subtract','multiply','divide']}})
    ]}},
    'outputs':{'type':'array','minItems':1,'maxItems':16,'items':obj({'label':STRING,'source':ID})}
})

def compile_plan(program,catalogs):
    """Execute only dependencies of requested outputs; discard unused work."""
    validate(program,SCHEMA)
    steps={s['id']:s for s in program['steps']}
    if len(steps)!=len(program['steps']): raise ValueError('Duplicate step IDs.')
    needed=set(); active=set()
    def visit(name):
        if name not in steps: raise ValueError('Unknown output/dependency reference: '+name)
        if name in active: raise ValueError('Cyclic plan dependency: '+name)
        if name in needed: return
        active.add(name); step=steps[name]
        for key in ('source','left','right'):
            if key in step: visit(step[key])
        active.remove(name); needed.add(name)
    for output in program['outputs']: visit(output['source'])
    compiled={**program,'steps':[s for s in program['steps'] if s['id'] in needed]}
    validate_plan(compiled,catalogs)
    return compiled

def validate_plan(program, catalogs):
    """Resolve fields, dependencies, types and units before any case request."""
    validate(program,SCHEMA)
    types={}
    for step in program['steps']:
        name=step['id']; op=step['op']
        if name in types: raise ValueError('Duplicate step ID: '+name)
        for rule in step.get('filters',[]):
            if rule['op']=='in' and (not isinstance(rule['value'],list) or not rule['value']): raise ValueError('in requires a nonempty array.')
            if rule['op']!='in' and isinstance(rule['value'],list): raise ValueError('Only in accepts array values.')
            if rule['op'] in ('lt','le','gt','ge'): finite(rule['value'])
        if op=='select':
            kind=step['object_type']; catalog=catalogs.get(kind,[])
            fields={f['variable_name']:f for f in catalog}
            required=set(step['fields']) | set(IDENTITIES[kind]) | {f['field'] for f in step.get('filters',[])}
            match=step.get('match')
            if match:
                org={f['variable_name'] for f in organizational_fields(catalog)}
                selected=set(match.get('fields',org))
                if not selected or not selected<=org: raise ValueError('Match fields must be organizational fields from the catalog.')
                required|=selected
            unknown=required-set(fields)
            if unknown: raise ValueError('Unknown '+kind+' fields: '+', '.join(sorted(unknown))+'. Inspect fields before revising the plan.')
            types[name]={'kind':'selection','object_type':kind,'fields':required}
        elif op=='filter':
            source=types.get(step['source'],{})
            if source.get('kind')!='selection': raise ValueError(step['id']+': filter source '+step['source']+' must reference a selection, not a quantity. Outputs can reference quantities directly.')
            if not {f['field'] for f in step['filters']}<=source['fields']: raise ValueError('Filter uses fields absent from its selection.')
            types[name]=source
        elif op=='aggregate':
            source=types.get(step['source'],{})
            if source.get('kind')!='selection': raise ValueError(step['id']+': aggregate source '+step['source']+' must reference a selection. Outputs can reference quantities directly; no report step is needed.')
            if step['method']=='count': unit='count'
            else:
                field=step.get('field')
                if field not in source['fields']: raise ValueError('Aggregate field is absent from its selection.')
                unit=catalog_unit(catalogs[source['object_type']],field) or 'unknown'
                if step.get('assumed_unit'):
                    if unit!='unknown' and step['assumed_unit']!=unit: raise ValueError('Assumed unit contradicts catalog evidence for '+field)
                    if unit=='unknown' and not step.get('assumption'): raise ValueError('An unknown source unit requires an explicit assumption explanation.')
                    unit=step['assumed_unit']
            types[name]={'kind':'quantity','unit':unit}
        else:
            refs=[step['source']] if op=='convert' else [step['left'],step['right']]
            if any(types.get(r,{}).get('kind')!='quantity' for r in refs): raise ValueError('Arithmetic and conversion require earlier quantities.')
            if op=='convert':
                convert(1,types[refs[0]]['unit'],step['unit'],conversion_pf(step,types[refs[0]]['unit']))
                unit=step['unit']
            else: unit=calculation_unit(step['method'],types[refs[0]]['unit'],types[refs[1]]['unit'])
            types[name]={'kind':'quantity','unit':unit}
    for output in program['outputs']:
        if types.get(output['source'],{}).get('kind')!='quantity': raise ValueError('Report outputs must reference computed quantities.')
    return types

def calculation_unit(method,left,right):
    if 'unknown' in (left,right): raise ValueError('Arithmetic requires established source units.')
    if method in ('add','subtract'):
        if left!=right: raise ValueError('Addition/subtraction require matching units; convert first.')
        return left
    if method=='divide': return 'ratio' if left==right else left+'/'+right
    return left+' × '+right

def conversion_pf(step,source_unit):
    pf=step.get('power_factor')
    # A redundant unity factor cannot change an ordinary scale conversion.
    # Non-unity factors and incompatible dimensions remain validation errors.
    if pf==1 and any(source_unit in group and step['unit'] in group for group in
                    ({'MW','GW'},{'MVA','GVA'},{'miles','km'})): return None
    return pf

def finite(value):
    if isinstance(value,bool): raise ValueError('Boolean is not a numeric measurement.')
    result=float(value)
    if not math.isfinite(result): raise ValueError('Measurement is not finite.')
    return result

def accepts(row, rules):
    for rule in rules:
        actual=row[rule['field']]; expected=rule['value']; op=rule['op']
        if op in ('lt','le','gt','ge'):
            a,b=finite(actual),finite(expected)
            ok={'lt':a<b,'le':a<=b,'gt':a>b,'ge':a>=b}[op]
        else:
            def equal(a,b):
                if isinstance(b,(float,int)) and not isinstance(b,bool): return finite(a)==finite(b)
                return str(a).strip().casefold()==str(b).strip().casefold()
            if op=='in':
                if not isinstance(expected,list): raise ValueError('in requires an array.')
                ok=any(equal(actual,v) for v in expected)
            elif op=='contains': ok=str(expected).casefold() in str(actual).casefold()
            else: ok=equal(actual,expected) if op=='eq' else not equal(actual,expected)
        if not ok: return False
    return True

class Executor:
    def __init__(self,program,catalogs,call,directory,save,progress):
        self.program=program; self.catalogs=catalogs; self.call=call; self.directory=Path(directory)
        self.types=validate_plan(program,catalogs); self.save=save; self.progress=progress
        self.directory.mkdir(parents=True,exist_ok=True)

    def rows(self,selection):
        return json.loads(Path(selection['path']).read_text(encoding='utf-8'))

    def selection(self,rows,kind,notes,key,unresolved=False):
        path=self.directory/(key+'.json'); write_json(path,rows)
        return {'kind':'selection','path':str(path),'object_type':kind,'count':len(rows),'notes':notes,'unresolved':unresolved}

    async def execute(self,case,step,values,key):
        op=step['op']; notes=[]
        if op=='select':
            kind=step['object_type']; fields=sorted(self.types[step['id']]['fields'])
            result=unpack(await self.call({'case_path':case,'object_type':kind,'fields':fields,'constraints':[],'limit':1}))
            if result.get('artifact_complete') is not True: raise ValueError('Query did not produce a complete export.')
            aliases={f['variable_name']:f.get('concise_name') for f in self.catalogs[kind]}
            seen={}; count=0
            with Path(result['artifact_path']).open(encoding='utf-8-sig',newline='') as stream:
                for _ in range(int(result.get('artifact_skip_rows',0))): next(stream)
                reader=csv.DictReader(stream)
                headers={f:next((h for h in (f,aliases.get(f)) if h in (reader.fieldnames or [])),None) for f in fields}
                if not all(headers.values()): raise ValueError('Required columns missing from complete export.')
                for raw in reader:
                    check_stopped(); count+=1; row={f:raw[h] for f,h in headers.items()}
                    identity=tuple(row[f] for f in IDENTITIES[kind])
                    if any(not v.strip() for v in identity): raise ValueError('Missing object identity.')
                    if identity in seen and seen[identity]!=row: raise ValueError('Conflicting duplicate object identity.')
                    seen[identity]=row
            if count!=result['artifact_row_count']: raise ValueError('Export row count mismatch.')
            rows=list(seen.values()); unresolved=False
            if step.get('match'):
                match=step['match']; org=match.get('fields') or [f['variable_name'] for f in organizational_fields(self.catalogs[kind])]
                rows=[r for r in rows if any(matches_company(r[f],match['text']) for f in org)]
                evidence=sorted({f+'='+r[f] for r in rows for f in org if matches_company(r[f],match['text'])})
                notes=['Organizational match: '+('; '.join(evidence[:20]) if evidence else 'unresolved; no matching tags')]
                unresolved=not rows
                notes.append('Area/zone membership does not establish ownership or operation.')
            rows=[r for r in rows if accepts(r,step.get('filters',[]))]
            return self.selection(rows,kind,notes,key,unresolved)
        source=values.get(step.get('source'))
        if op=='filter': return self.selection([r for r in self.rows(source) if accepts(r,step['filters'])],source['object_type'],source['notes'],key,source['unresolved'])
        unit=self.types[step['id']]['unit']
        if op=='aggregate':
            rows=self.rows(source); notes=list(source['notes']); method=step['method']
            if step.get('assumption'): notes.append('Assumption: '+step['assumption'])
            if source['unresolved']: value=None
            elif method=='count': value=len(rows)
            else:
                nums=[finite(r[step['field']]) for r in rows]
                value=(sum(nums) if method=='sum' else (sum(nums)/len(nums) if method=='mean' else min(nums) if method=='min' else max(nums)) if nums else None)
            notes.append('Aggregate '+method+' of '+step.get('field','objects')+'; '+str(len(rows))+' unique objects.')
        elif op=='convert':
            notes=list(source['notes']); pf=conversion_pf(step,source['unit'])
            value=None if source['value'] is None else convert(source['value'],source['unit'],step['unit'],pf)['value']
            if pf is not None: notes.append('Power factor assumption: '+str(pf))
        else:
            left,right=values[step['left']],values[step['right']]; notes=list(dict.fromkeys(left['notes']+right['notes']))
            a,b=left['value'],right['value']
            if a is None or b is None: value=None
            elif step['method']=='divide' and b==0: raise ValueError('Division by zero.')
            else: value={'add':lambda:a+b,'subtract':lambda:a-b,'multiply':lambda:a*b,'divide':lambda:a/b}[step['method']]()
        if value is not None: finite(value)
        return {'kind':'quantity','value':value,'unit':unit,'notes':notes}

    async def run(self,cases,state):
        digest=hashlib.sha256(json.dumps([self.program,cases],sort_keys=True).encode()).hexdigest()
        if state.get('digest')!=digest: state.clear(); state.update(digest=digest,cases={})
        for index,case in enumerate(cases):
            record=state['cases'].setdefault(case,{'values':{},'status':'running'})
            if record['status'] in ('completed','failed'): continue
            for step in self.program['steps']:
                check_stopped()
                if step['id'] in record['values']: continue
                self.progress(index+1,len(cases),step['id']); self.save()
                try:
                    # Keep selection + atomic-write filenames below Windows MAX_PATH
                    # even under a chat's nested durable result directory.
                    key=hashlib.sha256((digest+case+step['id']).encode()).hexdigest()[:20]
                    record['values'][step['id']]=await self.execute(case,step,record['values'],key)
                    self.save()
                except PermissionError: raise
                except (ValueError,KeyError,TypeError,OSError,ZeroDivisionError) as exc:
                    record.update(status='failed',error=str(exc)); self.save(); break
            else: record['status']='completed'; self.save()
        return state

def render(program,state):
    def safe(v): return str(v).replace('|','\\|').replace('\n',' ')
    lines=['| Case | '+' | '.join(safe(o['label']) for o in program['outputs'])+' | Status |',
           '| --- | '+' | '.join('---:' for _ in program['outputs'])+' | --- |']
    notes=[]
    names=[case.replace('\\','/').rsplit('/',1)[-1] for case in state['cases']]
    for case,record in state['cases'].items():
        cells=[]
        for output in program['outputs']:
            quantity=record['values'].get(output['source'],{}); value=quantity.get('value')
            cells.append('Unknown' if value is None else f'{value:,.6g} '+safe(quantity['unit']))
            notes.extend(quantity.get('notes',[]))
        status=record['status'] if record['status']=='failed' else ('unresolved' if 'Unknown' in cells else 'completed')
        parts=case.replace('\\','/').split('/')
        display=parts[-1] if names.count(parts[-1])==1 else '/'.join(parts[-2:])
        lines.append('| '+safe(display)+' | '+' | '.join(cells)+' | '+status+' |')
        if record.get('error'): notes.append(case+': '+record['error'])
    return '\n'.join(lines+['']+['- '+safe(n) for n in dict.fromkeys(notes)])
