"""Unit provenance and explicit conversions for stored numeric observations."""
import math
import re


def source_unit(payload, path):
    """Prefer reported metadata; recognize explicit unit-bearing field labels only."""
    nodes=[payload]
    if isinstance(payload,dict) and isinstance(payload.get('structuredContent'),dict):
        nodes.append(payload['structuredContent'])
    parts=[p.replace('~1','/').replace('~0','~') for p in path.split('/')[1:]]
    field=parts[parts.index('numeric_columns')+1] if 'numeric_columns' in parts and parts.index('numeric_columns')+1<len(parts) else (parts[-1] if parts else '')
    for node in nodes:
        if not isinstance(node,dict):
            continue
        units=node.get('units',{})
        if isinstance(units,dict) and isinstance(units.get(path),str):
            return {'unit':units[path],'evidence':'tool unit metadata: '+path}
        columns=node.get('column_units',{})
        if isinstance(columns,dict) and isinstance(columns.get(field),str):
            return {'unit':columns[field],'evidence':'tool column_units: '+field}
    if field.lower() in ('miles','mile','km'):
        return {'unit':'miles' if field.lower()!='km' else 'km','evidence':'explicit source column: '+field}
    match=re.search(r'(?:^|[^A-Z])(MVA|GVA|MW|GW|kV)(?:\s*\))?$',field)
    if match:
        return {'unit':match.group(1),'evidence':'explicit source column suffix: '+field}
    return {'unit':'unknown','evidence':'No supported explicit unit metadata or unit-bearing source column.'}


def convert(value, source, target, power_factor=None):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Unit conversion requires a finite numeric value.')
    groups={'MW':('power',1e6),'GW':('power',1e9),'MVA':('apparent_power',1e6),
            'GVA':('apparent_power',1e9),'miles':('length',1609.344),'km':('length',1000)}
    if source not in groups or target not in groups:
        raise ValueError('Source or target unit is unknown or unsupported; inspect the tool unit report first.')
    source_kind,scale=groups[source]
    target_kind,target_scale=groups[target]
    assumption=None
    if source_kind!=target_kind:
        if source_kind!='apparent_power' or target_kind!='power':
            raise ValueError('These units are incompatible.')
        if type(power_factor) not in (int,float) or not 0<=power_factor<=1:
            raise ValueError('MVA/GVA to MW/GW requires an explicit power_factor between 0 and 1.')
        scale*=power_factor
        assumption={'power_factor':power_factor}
    elif power_factor is not None:
        raise ValueError('Power factor applies only to apparent-power to real-power conversion.')
    result=value*scale/target_scale
    if not math.isfinite(result):
        raise ValueError('Converted quantity must be finite.')
    return {'value':result,'unit':target,'source_value':value,'source_unit':source,'assumption':assumption}
