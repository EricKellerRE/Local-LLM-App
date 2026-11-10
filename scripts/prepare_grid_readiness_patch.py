"""Stage code-only Grid Workshop changes for review and installation.

Does not discover, open, or read any case or case-derived artifact.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT.parent / "Grid-Workshop" / "powerworld-aux-agent"
STAGE = ROOT / "readiness_patch"


def stage(relative, transform):
    source = GRID / relative
    target = STAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(transform(source.read_text(encoding="utf-8")), encoding="utf-8")


def models(text):
    start = text.index("class ResultsSummarizeArtifactRequest")
    end = text.index("class ResultsInspectArtifactRequest", start)
    section = text[start:end]
    section = section.replace('    delimiter: str | None = None\n', '    delimiter: str | None = None\n    skip_rows: int = 0\n    filters: list[dict[str, Any]] = field(default_factory=list)\n')
    section = section.replace('            "delimiter": {"type": ["string", "null"]},', '''            "skip_rows": {"type": "integer", "minimum": 0, "maximum": 100, "description": "CSV metadata lines before the header. PowerWorld SaveData exports use 1."},
            "filters": {
                "type": "array", "description": "AND filters applied to full CSV rows before aggregation; use discovered exact column names.",
                "items": {
                    "type": "object", "required": ["column", "op", "value"],
                    "properties": {
                        "column": {"type": "string"},
                        "op": {"type": "string", "enum": ["eq", "ne", "in", "lt", "le", "gt", "ge"]},
                        "value": {},
                        "case_sensitive": {"type": "boolean", "default": True}
                    }, "additionalProperties": False
                }
            },
            "delimiter": {"type": ["string", "null"]},''')
    return text[:start] + section + text[end:]


def sidecar(text):
    text = text.replace('import json\n', 'import json\nimport math\n', 1)
    marker = 'def summarize_csv_artifact(path: Path, arguments: dict[str, Any]) -> dict[str, Any]:'
    helper = '''def _csv_matches(row: dict[str, str], filters: list[dict[str, Any]]) -> bool:
    for rule in filters:
        actual = row.get(rule["column"], "")
        expected = rule["value"]
        op = rule["op"]
        if op in {"lt", "le", "gt", "ge"} or isinstance(expected, (int, float)):
            left = _coerce_float(actual)
            right = _coerce_float(expected)
            if left is None or right is None or not math.isfinite(left) or not math.isfinite(right):
                return False
        else:
            def normalize(value):
                value = str(value).strip()
                return value if rule.get("case_sensitive", True) else value.casefold()
            left = normalize(actual)
            right = [normalize(value) for value in expected] if op == "in" else normalize(expected)
        matched = {
            "eq": lambda: left == right, "ne": lambda: left != right,
            "in": lambda: left in right, "lt": lambda: left < right,
            "le": lambda: left <= right, "gt": lambda: left > right,
            "ge": lambda: left >= right,
        }[op]()
        if not matched:
            return False
    return True


'''
    text = text.replace(marker, helper + marker, 1)
    start = text.index(marker)
    end = text.index('def summarize_json_artifact', start)
    section = text[start:end]
    section = section.replace('    row_count = 0', '''    filters = arguments.get("filters") or []
    skip_rows = int(arguments.get("skip_rows", 0))
    if not 0 <= skip_rows <= 100:
        raise ValueError("skip_rows must be between 0 and 100")
    for rule in filters:
        if not isinstance(rule, dict) or not {"column", "op", "value"} <= rule.keys():
            raise ValueError("Each CSV filter requires column, op, and value")
        if rule["op"] not in {"eq", "ne", "in", "lt", "le", "gt", "ge"}:
            raise ValueError("Unsupported CSV filter operator")
        if rule["op"] == "in" and not isinstance(rule["value"], list):
            raise ValueError("The in filter requires an array value")
    source_row_count = 0
    group_numeric: dict[str, dict[str, dict[str, Any]]] = {}
    row_count = 0''', 1)
    section = section.replace('        reader = csv.DictReader(handle, delimiter=delimiter)', '''        for _ in range(skip_rows):
            next(handle, None)
        reader = csv.DictReader(handle, delimiter=delimiter)''', 1)
    section = section.replace('        for row in reader:', '''        requested = {rule["column"] for rule in filters} | {name for name in [metric, group_by] if name}
        unknown = requested - set(columns)
        if unknown:
            raise ValueError(f"Unknown CSV columns: {sorted(unknown)}; available: {columns}")
        for row in reader:
            source_row_count += 1''', 1)
    section = section.replace('            row_count += 1', '''            if not _csv_matches(clean_row, filters):
                continue
            row_count += 1''', 1)
    section = section.replace('                if numeric is None:', '                if numeric is None or not math.isfinite(numeric):', 1)
    section = section.replace('                stats["max"] = max(stats["max"], numeric)', '''                stats["max"] = max(stats["max"], numeric)
                if group_by:
                    group_stats = group_numeric.setdefault(clean_row.get(group_by, ""), {}).setdefault(
                        column, {"count": 0, "sum": 0.0, "min": numeric, "max": numeric}
                    )
                    group_stats["count"] += 1
                    group_stats["sum"] += numeric
                    group_stats["min"] = min(group_stats["min"], numeric)
                    group_stats["max"] = max(group_stats["max"], numeric)''', 1)
    section = section.replace('                if metric_value is not None:', '                if metric_value is not None and math.isfinite(metric_value):', 1)
    section = section.replace('        "row_count": row_count,', '''        "row_count": row_count,
        "source_row_count": source_row_count,
        "filtered_out_count": source_row_count - row_count,
        "skip_rows": skip_rows,
        "filters": filters,''', 1)
    section = section.replace('            "group_by": group_by,', '''            "group_by": group_by,
            "group_count": len(group_counts),
            "groups_truncated": len(group_counts) > top_n,
            "numeric_columns": {name: group_numeric.get(name, {}) for name, _ in group_counts.most_common(top_n)},''', 1)
    return text[:start] + section + text[end:]


stage("powerworld_aux_agent/tool_models.py", models)
stage("powerworld_aux_agent/results_sidecar.py", sidecar)
stage("powerworld_aux_agent/tool_registry.py", lambda text: text.replace(
    'Large-result deterministic summarizer. Use for CSV, JSON, or text artifacts',
    'Large-result deterministic summarizer. CSV supports conjunctive entity/voltage/status filters, skip_rows for metadata (PowerWorld SaveData: 1), and grouped numeric column sums with counts and explicit group truncation. Use for CSV, JSON, or text artifacts'))
print(f"Staged code-only patch in {STAGE}")
