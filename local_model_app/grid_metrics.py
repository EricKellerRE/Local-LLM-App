"""Fixed read-only grid metrics pipeline. The model supplies schema mappings only."""
import csv
import math
import re
from pathlib import Path

from .quantity_units import convert


def object_schema(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

FIELD={'type':'string','minLength':1}
FIELDS={'type':'array','items':FIELD,'minItems':1,'uniqueItems':True}
ATTRIBUTION={'type':'array','minItems':1,'items':object_schema({
    'column':FIELD,'op':{'type':'string','enum':['eq','in']},
    'value':{'anyOf':[{'type':'string'},{'type':'number'},{'type':'array','minItems':1,'items':{'type':['string','number']}}]}})}
PARAMETERS=object_schema({
    'case_paths':{'type':'array','minItems':1,'uniqueItems':True,'items':FIELD},
    'branch':object_schema({'identity_fields':FIELDS,'voltage_fields':FIELDS,
        'attribution':ATTRIBUTION,'length_field':FIELD,'length_unit':{'type':'string','enum':['miles','km']},
        'rating_field':FIELD,'rating_unit':{'type':'string','enum':['MVA','MW','GVA','GW']}}),
    'generator':object_schema({'identity_fields':FIELDS,'attribution':ATTRIBUTION,
        'capacity_field':FIELD,'capacity_unit':{'type':'string','enum':['MW','GW']}}),
    'attribution_and_units_evidence':{'type':'string','minLength':1},
})
SPEC={'type':'function','function':{'name':'host__run_grid_metrics',
    'description':'Run the entire requested 345 kV entity analysis deterministically for ALL discovered cases. Supply discovered canonical field mappings and explicit entity attribution. Host fixes voltage=345, queries full exports, uses returned filenames, deduplicates identities, sums complete data, converts units (PF=1 for MVA), and calculates GW-miles. Do not create arithmetic or artifact steps.',
    'parameters':PARAMETERS}}


def organizational_fields(catalog):
    """Catalog identifiers, not arbitrary measurement fields mentioning an area."""
    return [f for f in catalog if any(re.search(
        r'owner|operator|company|organi[sz]ation|(?:area|zone)(?:name|num|number)|^(?:area|zone)(?::\d+)?$',
        str(f.get(key) or ''),re.I) for key in ('variable_name','concise_name'))]


def catalog_unit(catalog, field):
    entry=next((f for f in catalog if f.get('variable_name')==field),{})
    text=' '.join(str(entry.get(k) or '') for k in ('concise_name','description'))
    units=set(re.findall(r'\b(MW|GW|MVA|GVA|km|miles)\b',text))
    # Canonical compact field labels such as MWMax remain explicit unit labels.
    label=str(entry.get('concise_name') or field)
    found=re.match(r'^(MW|GW|MVA|GVA)(?:Max|Min|$)',label)
    if found: units.add(found.group(1))
    if label.lower() in ('miles','km'): units.add(label.lower())
    return next(iter(units)) if len(units)==1 else None


def render_report(result):
    def cell(value): return str(value).replace('|','\\|').replace('\n',' ')
    def number(value): return f'{value:,.6f}'.rstrip('0').rstrip('.')
    lines=['Analysis completed for the following cases. Transmission covers branches whose two endpoints are 345 kV; generation uses installed maximum capacity.', '',
        '| Case | 345 kV miles | Transmission GW | Generation GW | GW-miles |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in result['cases']:
        name=row['case'].replace('\\','/').rsplit('/',1)[-1]
        lines.append('| '+cell(name)+' | '+' | '.join(number(row[key]) for key in ('miles_345kv','transmission_gw','generation_gw','gw_miles'))+' |')
    lines+=['','GW-miles is total 345 kV miles × total transmission GW. MVA ratings are converted using power factor 1.0; this is a rating-based measure, not actual power flow.','']
    for row in result['cases']:
        name=row['case'].replace('\\','/').rsplit('/',1)[-1]
        lines.append('**'+cell(name)+'**: '+str(row['branch_count'])+' matching branches; '+str(row['generator_count'])+' matching generators.')
        for label,rules in row['attribution_rules'].items():
            rendered=['`'+cell(r['column'])+'` '+('equals ' if r['op']=='eq' else 'is one of ')+cell(r['value']) for r in rules]
            lines.append('- '+label.capitalize()+' attribution: '+' AND '.join(rendered)+'.')
        units=row['source_units']
        lines.append('- Source units: length '+units['length']+', rating '+units['rating']+', capacity '+units['capacity']+'. Values above are converted to miles and GW.')
        lines.append('')
    return '\n'.join(lines).strip()


async def discover_organizations(cases, catalogs, call, checkpoint):
    inventories=[]
    for case in cases:
        for kind in ('Branch','Gen'):
            fields=organizational_fields(catalogs[kind])
            if not fields: raise ValueError('No organizational fields found in '+kind+' catalog.')
            checkpoint('organization_discovery')
            data=unpack(await call({'case_path':case,'object_type':kind,
                'fields':[f['variable_name'] for f in fields],'constraints':[],'limit':1}))
            if data.get('artifact_complete') is not True: raise ValueError('Organizational discovery requires a complete export.')
            values={f['variable_name']:set() for f in fields}
            with Path(data['artifact_path']).open(encoding='utf-8-sig',newline='') as stream:
                for _ in range(int(data.get('artifact_skip_rows',0))): next(stream)
                reader=csv.DictReader(stream)
                headers={f['variable_name']:next((n for n in (f['variable_name'],f.get('concise_name')) if n in (reader.fieldnames or [])),None) for f in fields}
                count=0
                for row in reader:
                    count+=1
                    for name,header in headers.items():
                        if header and str(row.get(header) or '').strip(): values[name].add(row[header].strip())
            if count!=data.get('artifact_row_count'): raise ValueError('Organizational export row count mismatch.')
            inventories.append({'case':case,'object_type':kind,'source_row_count':count,
                'tags':{name:sorted(items) for name,items in values.items() if items},
                'unavailable_fields':[name for name,header in headers.items() if header is None]})
    return inventories


def unpack(result):
    if result.get('isError'):
        raise ValueError(str(result.get('detail') or result.get('message') or 'Query failed'))
    value=result.get('structuredContent',result)
    if not isinstance(value,dict) or value.get('ok') is False:
        raise ValueError('Query did not return a successful structured export.')
    return value


def aggregate_export(result, identity_fields, metrics, catalog=()):
    data=unpack(result)
    if data.get('artifact_complete') is not True:
        raise ValueError('A complete filtered export is required; preview rows cannot be aggregated.')
    path=Path(data['artifact_path'])
    if path.suffix.lower()!='.csv': raise ValueError('Query export must be CSV.')
    totals={name:0.0 for name in metrics}
    seen={}; count=0
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for _ in range(int(data.get('artifact_skip_rows',0))): next(stream)
        reader=csv.DictReader(stream)
        def header(name):
            if name in (reader.fieldnames or []): return name
            aliases={f.get('concise_name') for f in catalog if f.get('variable_name')==name}
            matches=aliases & set(reader.fieldnames or [])
            if len(matches)==1: return matches.pop()
            return name
        identity_fields=[header(name) for name in identity_fields]
        metrics={key:header(name) for key,name in metrics.items()}
        required=set(identity_fields)|set(metrics.values())
        missing=required-set(reader.fieldnames or [])
        if missing:
            raise ValueError('Mapping must use exact exported headers. Missing: '+str(sorted(missing))+'; available: '+str(reader.fieldnames))
        for row in reader:
            count+=1
            identity=tuple(row[key].strip() for key in identity_fields)
            if any(not key for key in identity): raise ValueError('Missing asset identity; cannot safely deduplicate.')
            values={}
            for name,column in metrics.items():
                try: value=float(row[column].strip().replace(',',''))
                except (ValueError,TypeError): raise ValueError('Missing or nonnumeric measurement in '+column) from None
                if not math.isfinite(value) or value<0: raise ValueError('Invalid measurement in '+column)
                values[name]=value
            if identity in seen:
                if seen[identity]!=row: raise ValueError('Conflicting duplicate asset identity.')
                continue
            seen[identity]=row
            for name,value in values.items(): totals[name]+=value
    if count!=data.get('artifact_row_count'):
        raise ValueError('Export row count does not match query metadata.')
    return totals, len(seen)


async def run_metrics(mapping, call, checkpoint, catalogs=None):
    """No model actions occur inside this routine. Callback preserves approvals/cache."""
    reports=[]; catalogs=catalogs or {}
    branch=mapping['branch']; gen=mapping['generator']
    # Catalog evidence overrides a contradictory model-provided source unit.
    branch=dict(branch); gen=dict(gen)
    for section,kind,pairs in ((branch,'Branch',(('length_field','length_unit'),('rating_field','rating_unit'))),
                               (gen,'Gen',(('capacity_field','capacity_unit'),))):
        for field_key,unit_key in pairs:
            unit=catalog_unit(catalogs.get(kind,()),section[field_key])
            if unit: section[unit_key]=unit
    for case in mapping['case_paths']:
        checkpoint('branches')
        fields=list(dict.fromkeys([*branch['identity_fields'],*branch['voltage_fields'],
            branch['length_field'],branch['rating_field'],*[r['column'] for r in branch['attribution']]]))
        result=await call({'case_path':case,'object_type':'Branch','fields':fields,
            'constraints':[*branch['attribution'],*[{'column':f,'op':'eq','value':345} for f in branch['voltage_fields']]],'limit':1})
        checkpoint('branch_totals')
        totals,branches=aggregate_export(result,branch['identity_fields'],{'miles':branch['length_field'],'rating':branch['rating_field']},catalogs.get('Branch',()))
        if not branches:
            raise ValueError('No branches matched the proposed organizational attribution and voltage. Attribution is unresolved; this is not evidence of zero assets. Reconsider the observed organizational tags.')
        checkpoint('generators')
        fields=list(dict.fromkeys([*gen['identity_fields'],gen['capacity_field'],*[r['column'] for r in gen['attribution']]]))
        result=await call({'case_path':case,'object_type':'Gen','fields':fields,'constraints':gen['attribution'],'limit':1})
        generation,generators=aggregate_export(result,gen['identity_fields'],{'capacity':gen['capacity_field']},catalogs.get('Gen',()))
        if not generators:
            raise ValueError('No generators matched the proposed organizational attribution. Attribution is unresolved; this is not evidence of zero generation.')
        checkpoint('calculation')
        miles=convert(totals['miles'],branch['length_unit'],'miles')['value']
        pf=1.0 if branch['rating_unit'] in ('MVA','GVA') else None
        transmission=convert(totals['rating'],branch['rating_unit'],'GW',pf)['value']
        capacity=convert(generation['capacity'],gen['capacity_unit'],'GW')['value']
        reports.append({'case':case,'miles_345kv':miles,'transmission_gw':transmission,
            'generation_gw':capacity,'gw_miles':miles*transmission,'branch_count':branches,
            'generator_count':generators,'assumption':'MVA to GW uses PF=1; installed generation capacity.',
            'source_units':{'length':branch['length_unit'],'rating':branch['rating_unit'],'capacity':gen['capacity_unit']},
            'attribution_rules':{'branch':branch['attribution'],'generator':gen['attribution']},
            'attribution_evidence':mapping['attribution_and_units_evidence']})
    return {'cases':reports}
