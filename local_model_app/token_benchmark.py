from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_model_app.token_tuning import bracketed_budget_search


LONG_URL = (
    "https://example.org/research/articles/long-term-memory-mechanisms?"
    "study=protein-synthesis-and-synaptic-tagging&population=adult-mice&"
    "comparison=early-versus-late-consolidation&source=benchmark"
)


@dataclass(frozen=True)
class Fixture:
    name: str
    candidates: list[int]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    score: Callable[[dict[str, Any]], tuple[bool, str]]


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found.")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _choice(response: dict[str, Any]) -> dict[str, Any]:
    return dict(response["choices"][0]["message"])


def score_tool_url(response: dict[str, Any]) -> tuple[bool, str]:
    calls = _choice(response).get("tool_calls") or []
    if len(calls) != 1:
        return False, f"expected one tool call, received {len(calls)}"
    function = calls[0].get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False, "tool arguments were not valid JSON"
    passed = function.get("name") == "fetch_url" and arguments.get("url") == LONG_URL
    return passed, "exact long URL preserved" if passed else f"wrong call or URL: {function}"


def score_next_chunk(response: dict[str, Any]) -> tuple[bool, str]:
    calls = _choice(response).get("tool_calls") or []
    if len(calls) != 1:
        return False, f"expected one continuation call, received {len(calls)}"
    function = calls[0].get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False, "continuation arguments were not valid JSON"
    passed = (
        function.get("name") == "fetch_url" and arguments.get("url") == LONG_URL and
        int(arguments.get("start_index") or -1) == 30000
    )
    return passed, "continued at the exact cursor" if passed else f"wrong continuation: {function}"


def score_relevance(response: dict[str, Any]) -> tuple[bool, str]:
    try:
        payload = _json_object(str(_choice(response).get("content") or ""))
    except (ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
    selected = set(payload.get("keep_numbers") or [])
    ambiguous = set(payload.get("needs_abstract_numbers") or [])
    passed = selected == {1, 3} and not ambiguous
    return passed, f"kept={sorted(selected)}, needs_abstract={sorted(ambiguous)}"


def score_notes(response: dict[str, Any]) -> tuple[bool, str]:
    content = str(_choice(response).get("content") or "")
    required = [
        "https://example.org/source-a", "https://example.org/source-b",
        "method", "limitation", "protein",
    ]
    missing = [value for value in required if value.lower() not in content.lower()]
    return not missing, "all evidence fields retained" if not missing else f"missing={missing}"


def score_post_tool_summary(response: dict[str, Any]) -> tuple[bool, str]:
    message = _choice(response)
    if message.get("tool_calls"):
        return False, "called another tool after a complete result"
    content = str(message.get("content") or "")
    required = [LONG_URL, "protein", "limitation"]
    missing = [value for value in required if value.lower() not in content.lower()]
    return not missing, "complete result summarized with provenance" if not missing else f"missing={missing}"


def score_section(response: dict[str, Any]) -> tuple[bool, str]:
    content = str(_choice(response).get("content") or "")
    words = len(re.findall(r"\b[\w'-]+\b", content))
    urls = {url.rstrip(".,;)") for url in re.findall(r"https?://\S+", content)}
    passed = words >= 600 and len(urls) >= 2 and all(
        marker in content.lower() for marker in ("limitation", "causal", "conclusion")
    )
    return passed, f"words={words}, urls={len(urls)}"


def score_section_chunk(response: dict[str, Any]) -> tuple[bool, str]:
    content = str(_choice(response).get("content") or "")
    words = len(re.findall(r"\b[\w'-]+\b", content))
    urls = {url.rstrip(".,;)") for url in re.findall(r"https?://\S+", content)}
    passed = words >= 300 and len(urls) >= 2 and all(
        marker in content.lower() for marker in ("limitation", "causal", "conclusion")
    )
    return passed, f"words={words}, urls={len(urls)}"


def fetch_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a public webpage in chunks using an exact URL and optional character cursor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "start_index": {"type": "integer", "minimum": 0},
                    "max_length": {"type": "integer", "minimum": 1000},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    }


def fixtures(suite: str) -> list[Fixture]:
    tool = fetch_tool()
    selected = [
        Fixture(
            name="tool_long_url",
            candidates=[128, 192, 256, 384, 512, 768, 1024],
            messages=[
                {"role": "system", "content": "Call exactly one supplied tool. Do not describe the call."},
                {"role": "user", "content": f"Fetch this exact URL now: {LONG_URL}"},
            ],
            tools=[tool],
            score=score_tool_url,
        ),
        Fixture(
            name="reference_relevance",
            candidates=[128, 192, 256, 384, 512],
            messages=[
                {"role": "system", "content": (
                    "Select only candidates relevant to biological mechanisms of long-term memory. Return only "
                    "JSON with keep_numbers and needs_abstract_numbers. Use needs_abstract only when supplied context "
                    "is insufficient."
                )},
                {"role": "user", "content": json.dumps({"candidates": [
                    {"number": 1, "title": "Protein synthesis in long-term memory consolidation"},
                    {"number": 2, "title": "Seasonal precipitation forecasting"},
                    {"number": 3, "title": "Synaptic tagging and memory persistence review"},
                ]})},
            ],
            tools=[],
            score=score_relevance,
        ),
    ]
    if suite == "pilot":
        return selected
    selected.extend([
        Fixture(
            name="post_tool_summary",
            candidates=[256, 384, 512, 768, 1024, 1536, 2048],
            messages=[
                {"role": "system", "content": (
                    "The tool result is complete. Record its evidence concisely with the exact source URL, method, "
                    "finding, and limitation. Do not call another tool."
                )},
                {"role": "user", "content": "Extract the evidence needed for a later report."},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_fixture", "type": "function",
                    "function": {"name": "fetch_url", "arguments": {"url": LONG_URL}},
                }]},
                {"role": "tool", "tool_call_id": "call_fixture", "name": "fetch_url", "content": json.dumps({
                    "url": LONG_URL,
                    "content": (
                        "Method: post-training protein-synthesis inhibition in mice. Finding: later retention was "
                        "impaired. Limitation: systemic drug toxicity and off-target effects weaken specificity."
                    ),
                    "next_start": None,
                    "complete": True,
                })},
            ],
            tools=[tool],
            score=score_post_tool_summary,
        ),
        Fixture(
            name="source_notes",
            candidates=[512, 768, 1024, 1536],
            messages=[
                {"role": "system", "content": (
                    "Write concise source-specific evidence notes. Preserve each exact URL, method, mechanism, and "
                    "limitation. Do not merge the sources."
                )},
                {"role": "user", "content": json.dumps({"sources": [
                    {"id": "a", "url": "https://example.org/source-a", "content": (
                        "A mouse conditioning experiment inhibited protein synthesis after training. The method was "
                        "systemic anisomycin, which impaired later retention. Limitation: toxicity and off-target effects."
                    )},
                    {"id": "b", "url": "https://example.org/source-b", "content": (
                        "A correlational human imaging study associated hippocampal replay with retention. The method "
                        "was fMRI. Limitation: temporal resolution does not establish cellular causality."
                    )},
                ]})},
            ],
            tools=[],
            score=score_notes,
        ),
        Fixture(
            name="report_section_chunk",
            candidates=[512, 768, 1024],
            messages=[
                {"role": "system", "content": (
                    "Write one 300-450 word bounded segment of a longer report section from only the evidence supplied. "
                    "Explain causal strength, limitations, and a supported conclusion. Cite exact URLs beside claims."
                )},
                {"role": "user", "content": (
                    "Source A https://example.org/source-a: protein-synthesis inhibition after training impaired later "
                    "retention in mice; pharmacological off-target effects limit interpretation. Source B "
                    "https://example.org/source-b: human fMRI linked hippocampal replay to retention, but the method is "
                    "correlational and temporally coarse. Distinguish causal evidence from association."
                )},
            ],
            tools=[],
            score=score_section_chunk,
        ),
    ])
    return selected


class ApiClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def wait_ready(self, seconds: int) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        last_error = "not ready"
        while time.monotonic() < deadline:
            try:
                health = self.get("/health")
                if health.get("status") == "ready":
                    return health
                last_error = str(health)
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"Local Model did not become ready: {last_error}")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_benchmark(
    client: ApiClient,
    selected_fixtures: list[Fixture],
    output_path: Path,
    *,
    repeats: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": client.get("/health").get("model"),
        "repeats": repeats,
        "fixtures": [],
    }
    _atomic_write(output_path, report)
    for fixture in selected_fixtures:
        evaluations: list[dict[str, Any]] = []

        def evaluate(budget: int) -> bool:
            started = time.monotonic()
            try:
                response = client.post("/v1/chat/completions", {
                    "messages": fixture.messages,
                    "tools": fixture.tools,
                    "temperature": 0,
                    "max_tokens": budget,
                })
                passed, detail = fixture.score(response)
                error = None
            except Exception as exc:
                passed, detail, error = False, "request failed", f"{type(exc).__name__}: {exc}"
            row = {
                "budget": budget,
                "passed": passed,
                "detail": detail,
                "error": error,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            evaluations.append(row)
            print(json.dumps({"fixture": fixture.name, **row}, ensure_ascii=False), flush=True)
            return passed

        result = bracketed_budget_search(fixture.candidates, evaluate, repeats=repeats)
        report["fixtures"].append({
            "name": fixture.name,
            "search": result.as_dict(),
            "evaluations": evaluations,
        })
        _atomic_write(output_path, report)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(item["search"]["selected_budget"] is not None for item in report["fixtures"])
    _atomic_write(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local-model token-budget boundary tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--suite", choices=["pilot", "full"], default="full")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--only", default="", help="Comma-separated fixture names to run.")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/token-budget-latest.json"))
    arguments = parser.parse_args()
    client = ApiClient(arguments.base_url, arguments.request_timeout)
    health = client.wait_ready(arguments.wait_seconds)
    print(json.dumps({"event": "ready", **health}), flush=True)
    selected = fixtures(arguments.suite)
    if arguments.only.strip():
        names = {value.strip() for value in arguments.only.split(",") if value.strip()}
        selected = [fixture for fixture in selected if fixture.name in names]
        missing = names - {fixture.name for fixture in selected}
        if missing:
            parser.error(f"Unknown fixture(s): {', '.join(sorted(missing))}")
    report = run_benchmark(
        client,
        selected,
        arguments.output.resolve(),
        repeats=max(1, arguments.repeats),
    )
    print(json.dumps({
        "event": "complete",
        "passed": report["passed"],
        "output": str(arguments.output.resolve()),
    }), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
