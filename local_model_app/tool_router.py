from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from local_model_app.mcp_plugins import DiscoveredTool


DEFAULT_SCHEMA_BATCH = 4
MAX_ROUTED_SCHEMAS = 8
DEFAULT_SEMANTIC_WEIGHT = 36.0
LEXICAL_NAME_WEIGHT = 2.0
LEXICAL_TITLE_WEIGHT = 2.0
LEXICAL_DESCRIPTION_WEIGHT = 3.0
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "get", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "the",
    "this", "to", "use", "with", "you",
}
_SELF_CONTAINED_PREFIX = re.compile(
    r"^\s*(?:explain|define|teach me|what does\b.+\bmean|write|draft|compose|rewrite)\b",
    re.IGNORECASE,
)
_EXTERNAL_CONTEXT = re.compile(
    r"\b(?:attached|case|file|folder|artifact|result|job|server|calendar|email|database|repository|"
    r"spreadsheet|document|record|current|latest|these|this)\b|[a-z]:\\|https?://",
    re.IGNORECASE,
)
_BROAD_DESTRUCTIVE_REQUEST = re.compile(
    r"\b(?:delete|erase|wipe|destroy|remove)\b.*\b(?:all|every|computer|disk|drive|system|workstation)\b|"
    r"\b(?:all|every|computer|disk|drive|system|workstation)\b.*\b(?:delete|erase|wipe|destroy|remove)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    words = set(_TOKEN_PATTERN.findall(text.lower().replace("_", " ").replace("-", " ")))
    expanded = set(words)
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            expanded.add(f"{word[:-3]}y")
        elif len(word) > 3 and word.endswith("s"):
            expanded.add(word[:-1])
    return expanded - _STOP_WORDS


def _positive_description(description: str) -> str:
    negative_prefixes = (
        "do not ", "don't ", "never ", "avoid ", "not for ",
        "not intended ", "must not ", "should not ",
    )
    sentences = re.split(r"(?<=[.!?])\s+", description)
    positive = [
        sentence
        for sentence in sentences
        if not sentence.strip().lower().startswith(negative_prefixes)
    ]
    return " ".join(positive)


def _metadata_text(tool: DiscoveredTool) -> str:
    name = " ".join(_TOKEN_PATTERN.findall(tool.native_name.lower().replace("_", " ").replace("-", " ")))
    # Many well-described MCP tools append standardized prerequisite/result/state
    # sections. They remain available to lexical routing and the planner, but their
    # repeated wording can swamp the capability's distinguishing intent embedding.
    intent = re.split(
        r"\b(?:Prerequisites|Result|State):",
        _positive_description(tool.description),
        maxsplit=1,
    )[0]
    return ". ".join(part.strip() for part in (tool.title or "", name, intent) if part.strip())


def _self_contained_no_tool_hint(query: str) -> bool:
    """Recognize requests normally answerable by the model without external state."""
    self_contained = _SELF_CONTAINED_PREFIX.search(query) and not _EXTERNAL_CONTEXT.search(query)
    return bool(self_contained or _BROAD_DESTRUCTIVE_REQUEST.search(query))


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class TextEncoder(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingEncoder:
    """Lazy, local Hugging Face sentence encoder for compact metadata routing."""

    def __init__(self, model_id: str, *, device: str = "cpu", max_length: int = 256) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        return self.model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.to(torch.device(self.device))
        self._model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().tolist())
        return vectors

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if "e5" in self.model_id.lower():
            texts = [f"passage: {text}" for text in texts]
        return self.encode(texts)

    def encode_query(self, text: str) -> list[float]:
        model_name = self.model_id.lower()
        if "e5" in model_name:
            text = f"query: {text}"
        elif "bge" in model_name:
            text = f"Represent this sentence for searching relevant passages: {text}"
        vectors = self.encode([text])
        return vectors[0] if vectors else []


@dataclass(frozen=True)
class RoutedCatalog:
    ranked: tuple[DiscoveredTool, ...]
    scores: tuple[float, ...]
    lexical_scores: tuple[float, ...] = ()
    semantic_scores: tuple[float, ...] = ()
    semantic_backend: str | None = None
    semantic_error: str | None = None
    no_tool_hint: bool = False
    batch_size: int = DEFAULT_SCHEMA_BATCH
    limit: int = DEFAULT_SCHEMA_BATCH

    @property
    def visible(self) -> list[DiscoveredTool]:
        return list(self.ranked[: self.limit])

    @property
    def has_more(self) -> bool:
        return self.limit < min(len(self.ranked), MAX_ROUTED_SCHEMAS)

    @property
    def low_confidence(self) -> bool:
        if not self.scores or self.no_tool_hint:
            return True
        top_lexical = self.lexical_scores[0] if self.lexical_scores else self.scores[0]
        lexical_gap = top_lexical - (self.lexical_scores[1] if len(self.lexical_scores) > 1 else 0.0)
        if not self.semantic_scores:
            return top_lexical <= 0 or top_lexical < 5 or lexical_gap < 2
        top_semantic = self.semantic_scores[0]
        semantic_gap = top_semantic - (self.semantic_scores[1] if len(self.semantic_scores) > 1 else 0.0)
        lexical_confident = top_lexical >= 5 and lexical_gap >= 2 and top_semantic >= 0.12
        semantic_confident = top_semantic >= 0.48 and semantic_gap >= 0.035
        return not (lexical_confident or semantic_confident)

    def expanded(self) -> "RoutedCatalog":
        return RoutedCatalog(
            ranked=self.ranked,
            scores=self.scores,
            lexical_scores=self.lexical_scores,
            semantic_scores=self.semantic_scores,
            semantic_backend=self.semantic_backend,
            semantic_error=self.semantic_error,
            no_tool_hint=self.no_tool_hint,
            batch_size=self.batch_size,
            limit=min(self.limit + self.batch_size, len(self.ranked), MAX_ROUTED_SCHEMAS),
        )


class ToolRouter:
    """Rank a complete MCP catalog without copying tool schemas into model context."""

    def __init__(self, encoder: TextEncoder | None = None, *, semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT) -> None:
        self.encoder = encoder
        self.semantic_weight = max(0.0, semantic_weight)
        self._catalog_key: str | None = None
        self._catalog_vectors: list[list[float]] = []
        self._semantic_error: str | None = None

    @staticmethod
    def compact_catalog(tools: list[DiscoveredTool]) -> str:
        rows = []
        for tool in tools:
            metadata = ": ".join(part for part in (tool.title, tool.description) if part) or tool.kind
            rows.append(f"- {tool.exposed_name}: {metadata[:180]}")
        return "\n".join(rows)

    @staticmethod
    def _lexical_score(
        query_tokens: set[str],
        query_text: str,
        tool: DiscoveredTool,
        inverse_document_frequency: dict[str, float],
    ) -> float:
        name_tokens = _tokens(f"{tool.exposed_name} {tool.native_name}")
        title_tokens = _tokens(tool.title or "")
        description_tokens = _tokens(_positive_description(tool.description))

        def weighted_overlap(tokens: set[str]) -> float:
            return sum(inverse_document_frequency.get(token, 0.0) for token in query_tokens & tokens)

        value = (
            LEXICAL_NAME_WEIGHT * weighted_overlap(name_tokens)
            + LEXICAL_TITLE_WEIGHT * weighted_overlap(title_tokens)
            + LEXICAL_DESCRIPTION_WEIGHT * weighted_overlap(description_tokens)
        )
        native_phrase = " ".join(
            _TOKEN_PATTERN.findall(tool.native_name.lower().replace("_", " ").replace("-", " "))
        )
        if native_phrase and native_phrase in query_text:
            value += 4.0
        return value

    @staticmethod
    def _inverse_document_frequency(tools: list[DiscoveredTool]) -> dict[str, float]:
        document_frequency: dict[str, int] = {}
        for tool in tools:
            tokens = _tokens(
                f"{tool.exposed_name} {tool.native_name} {tool.title or ''} "
                f"{_positive_description(tool.description)}"
            )
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        catalog_size = len(tools)
        return {
            token: math.log((catalog_size + 1) / (frequency + 1))
            for token, frequency in document_frequency.items()
        }

    @staticmethod
    def _catalog_fingerprint(tools: list[DiscoveredTool]) -> str:
        material = "\n".join(f"{tool.exposed_name}\0{_metadata_text(tool)}" for tool in tools)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _semantic_scores(
        self,
        query: str,
        tools: list[DiscoveredTool],
        context: str | None = None,
    ) -> tuple[list[float], str | None]:
        if self.encoder is None or not tools:
            return [], None
        try:
            key = self._catalog_fingerprint(tools)
            if key != self._catalog_key:
                encode_documents = getattr(self.encoder, "encode_documents", self.encoder.encode)
                self._catalog_vectors = encode_documents([_metadata_text(tool) for tool in tools])
                if len(self._catalog_vectors) != len(tools):
                    raise ValueError("Semantic encoder returned the wrong number of catalog vectors")
                self._catalog_key = key
            encode_query = getattr(self.encoder, "encode_query", None)
            query_vector = encode_query(query) if callable(encode_query) else self.encoder.encode([query])[0]
            if not query_vector:
                raise ValueError("Semantic encoder returned no query vector")
            if context:
                context_vector = (
                    encode_query(context) if callable(encode_query) else self.encoder.encode([context])[0]
                )
                if len(context_vector) == len(query_vector):
                    blended = [0.8 * current + 0.2 * prior for current, prior in zip(query_vector, context_vector)]
                    norm = math.sqrt(sum(value * value for value in blended))
                    if norm:
                        query_vector = [value / norm for value in blended]
            self._semantic_error = None
            return [_cosine(query_vector, vector) for vector in self._catalog_vectors], None
        except Exception as exc:
            self._semantic_error = f"{type(exc).__name__}: {exc}"
            return [], self._semantic_error

    def route(
        self,
        query: str,
        tools: list[DiscoveredTool],
        *,
        context: str | None = None,
        batch_size: int = DEFAULT_SCHEMA_BATCH,
    ) -> RoutedCatalog:
        query_tokens = _tokens(query)
        query_text = " ".join(_TOKEN_PATTERN.findall(query.lower().replace("_", " ").replace("-", " ")))
        context_tokens = _tokens(context or "")
        context_text = " ".join(
            _TOKEN_PATTERN.findall((context or "").lower().replace("_", " ").replace("-", " "))
        )
        inverse_document_frequency = self._inverse_document_frequency(tools)
        lexical = [
            self._lexical_score(query_tokens, query_text, tool, inverse_document_frequency)
            + 0.2 * self._lexical_score(context_tokens, context_text, tool, inverse_document_frequency)
            for tool in tools
        ]
        semantic, semantic_error = self._semantic_scores(query, tools, context)

        rows = []
        for index, tool in enumerate(tools):
            semantic_score = max(0.0, semantic[index]) if semantic else 0.0
            combined = lexical[index] + self.semantic_weight * semantic_score
            rows.append((combined, lexical[index], semantic_score, tool))
        rows.sort(key=lambda row: (-row[0], row[3].exposed_name))
        return RoutedCatalog(
            ranked=tuple(row[3] for row in rows),
            scores=tuple(row[0] for row in rows),
            lexical_scores=tuple(row[1] for row in rows),
            semantic_scores=tuple(row[2] for row in rows) if semantic else (),
            semantic_backend=self.encoder.name if semantic and self.encoder is not None else None,
            semantic_error=semantic_error,
            no_tool_hint=_self_contained_no_tool_hint(query),
            batch_size=max(1, batch_size),
            limit=min(max(1, batch_size), len(rows)),
        )
