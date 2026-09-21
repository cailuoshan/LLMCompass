"""Stable JSON protocol between LLMCompass and an external search provider.

This module intentionally knows nothing about ``funcs`` or the transfer model.
"""

import hashlib
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


_problem_context = ContextVar("operator_problem", default={})


def stable_id(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def operator_problem(graph):
    token = _problem_context.set(graph)
    try:
        yield
    finally:
        _problem_context.reset(token)


def execute_provider_search(provider, name, stage, hardware, records, evaluate):
    """Support online probes and legacy rank-only providers without re-evaluation."""
    top_k = getattr(provider, "top_k", 1)
    measured = {}
    def callback(cid):
        if cid not in {r["candidate_id"] for r in records}:
            raise ProviderProtocolError("provider evaluated an unknown candidate")
        if cid not in measured:
            latency = float(evaluate(cid))
            if not math.isfinite(latency) or latency <= 0:
                raise ProviderProtocolError("candidate latency must be finite and positive")
            measured[cid] = latency
        return measured[cid]
    search = getattr(provider, "search_and_evaluate", None)
    if callable(search):
        ids = search(name, stage, hardware or {}, {}, records, top_k, callback)
        validate_ranked_ids(records, ids, len(records))
    else:
        ids = provider.rank(name, stage, hardware or {}, {}, records, top_k)
        validate_ranked_ids(records, ids, top_k)
    for cid in ids:
        callback(cid)
    return measured


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError):
            pass
    return value


def mapping_to_dict(mapping: Any) -> Dict[str, Any]:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return {str(key): _json_value(value) for key, value in mapping.items()}
    return {
        str(key): _json_value(value)
        for key, value in vars(mapping).items()
        if not str(key).startswith("_")
    }


def candidate_id(operator_name: str, execution_kind: str, strategy: Optional[str], mapping: Dict[str, Any], problem_id: str = "") -> str:
    payload = {
        "problem_id": problem_id,
        "operator_name": operator_name,
        "execution_kind": execution_kind,
        "strategy": strategy,
        "mapping": mapping_to_dict(mapping),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class MappingCandidate:
    operator_name: str
    operator_type: str
    execution_kind: str
    strategy: Optional[str]
    graph: Dict[str, Any]
    mapping: Dict[str, Any]
    mapping_object: Any = field(default=None, repr=False, compare=False)
    candidate_id: str = ""
    problem_id: str = ""
    search_id: str = ""

    def __post_init__(self):
        from software_model.design_point_recorder import get_trial_latency_modifiers
        context = _problem_context.get()
        self.graph = {**context, **self.graph}
        if context:
            self.graph["problem"] = context
        modifiers = get_trial_latency_modifiers()
        self.graph.setdefault("latency_multiplier", modifiers["multiplier"])
        self.graph.setdefault("latency_additive_s", modifiers["additive_s"])
        # Search identity uses the outer problem; alternative BMM lowerings share it.
        semantic = self.graph.get("problem") or self.graph
        pid = stable_id({"operator_type": semantic.get("operator_type", self.operator_type), "graph": semantic})
        if self.problem_id and self.problem_id != pid:
            raise ValueError("problem_id does not match graph")
        self.problem_id = pid
        sid = stable_id([self.operator_name, pid])
        if self.search_id and self.search_id != sid:
            raise ValueError("search_id does not match problem")
        self.search_id = sid
        expected = candidate_id(self.operator_name, self.execution_kind, self.strategy, self.mapping,
                                stable_id([pid, self.graph, self.operator_type]))
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("candidate_id does not match candidate fields")
        self.candidate_id = expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "search_id": self.search_id,
            "operator_name": self.operator_name,
            "operator_type": self.operator_type,
            "execution_kind": self.execution_kind,
            "strategy": self.strategy,
            "graph": _json_value(self.graph),
            "mapping": mapping_to_dict(self.mapping),
        }


@dataclass
class TrialResult:
    candidate_id: str
    local_latency_s: float
    measured: bool = True
    cycle_count: Optional[int] = None
    raw_local_latency_s: Optional[float] = None
    strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "local_latency_s": float(self.local_latency_s),
            "measured": bool(self.measured),
            "cycle_count": self.cycle_count,
            "raw_local_latency_s": self.raw_local_latency_s,
            "strategy": self.strategy,
        }


class ProviderProtocolError(ValueError):
    pass


def validate_ranked_ids(candidates: Iterable[Dict[str, Any]], ranked_ids: Iterable[str], top_k: Any) -> List[str]:
    records = list(candidates)
    if not records:
        raise ProviderProtocolError("candidate set is empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ProviderProtocolError("top_k must be a positive integer")
    ids = [record.get("candidate_id") for record in records]
    if any(not value for value in ids):
        raise ProviderProtocolError("candidate is missing candidate_id")
    if len(set(ids)) != len(ids):
        raise ProviderProtocolError("candidate set contains duplicate candidate_id")
    if ranked_ids is None:
        raise ProviderProtocolError("provider returned no candidate ids")
    selected = list(ranked_ids)
    if len(set(selected)) != len(selected):
        raise ProviderProtocolError("provider returned duplicate candidate ids")
    unknown = [value for value in selected if value not in set(ids)]
    if unknown:
        raise ProviderProtocolError("provider returned unknown candidate ids: {}".format(unknown))
    if len(selected) > top_k:
        raise ProviderProtocolError("provider returned {} ids for top_k={}".format(len(selected), top_k))
    if not selected:
        raise ProviderProtocolError("provider returned an empty ranking")
    return selected


def deduplicate_candidates(candidates: Iterable[MappingCandidate]) -> List[MappingCandidate]:
    """Remove equivalent candidates while preserving deterministic enum order.

    Candidate generation may reach the same mapping through different factor
    values, for example ``ceil(16 / 16) == ceil(16 / 32) == 1``.  Deduplication
    belongs at the generation boundary; protocol validation must still reject
    duplicates returned by an external provider.
    """
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        unique.append(candidate)
    return unique
