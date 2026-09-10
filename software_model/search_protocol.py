"""Stable JSON protocol between LLMCompass and an external search provider."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union


Strategy = Optional[Union[str, Dict[str, Any]]]


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
    return {str(key): _json_value(value) for key, value in vars(mapping).items() if not str(key).startswith("_")}


def candidate_id(operator_name: str, execution_kind: str, strategy: Strategy, mapping: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "operator_name": operator_name,
        "execution_kind": execution_kind,
        "strategy": strategy,
        "mapping": mapping_to_dict(mapping),
    }
    if metadata:
        payload["metadata"] = _json_value(metadata)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class MappingCandidate:
    operator_name: str
    operator_type: str
    execution_kind: str
    strategy: Strategy
    graph: Dict[str, Any]
    mapping: Dict[str, Any]
    mapping_object: Any = field(default=None, repr=False, compare=False)
    candidate_id: str = ""
    precision: Dict[str, Any] = field(default_factory=dict)
    layout: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("graph", "mapping", "precision", "layout", "resource_requirements"):
            if not isinstance(getattr(self, name), dict):
                raise ProviderProtocolError("candidate {} must be a dictionary".format(name))
        metadata = {key: value for key, value in {
            "precision": self.precision,
            "layout": self.layout,
            "resource_requirements": self.resource_requirements,
        }.items() if value}
        expected = candidate_id(self.operator_name, self.execution_kind, self.strategy, self.mapping, metadata)
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("candidate_id does not match candidate fields")
        self.candidate_id = expected

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MappingCandidate":
        data = dict(value)
        data.pop("schema_version", None)
        legacy = any(key in data for key in ("resource_hint", "validity", "conditions", "valid", "invalid_reason"))
        if "resource_requirements" not in data:
            data["resource_requirements"] = data.get("resource_hint", {})
        data.pop("resource_hint", None)
        for key in ("validity", "conditions", "valid", "invalid_reason"):
            data.pop(key, None)
        if legacy:
            data.pop("candidate_id", None)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "operator_name": self.operator_name,
            "operator_type": self.operator_type,
            "execution_kind": self.execution_kind,
            "strategy": _json_value(self.strategy),
            "graph": _json_value(self.graph),
            "mapping": mapping_to_dict(self.mapping),
            "precision": _json_value(self.precision),
            "layout": _json_value(self.layout),
            "resource_requirements": _json_value(self.resource_requirements),
        }


@dataclass
class TrialResult:
    candidate_id: str
    local_latency_s: float
    measured: bool = True
    cycle_count: Optional[int] = None
    raw_local_latency_s: Optional[float] = None
    strategy: Strategy = None

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


class SearchableOperator:
    """Common candidate interface exposed by software operators."""

    def enumerate_transfer_candidates(self, pcb_module, generation_mode="heuristic-GPU"):
        raise NotImplementedError

    def evaluate_transfer_candidate(self, pcb_module, candidate, trial_sink=None):
        raise NotImplementedError

    @staticmethod
    def validate_candidate(candidate: MappingCandidate) -> MappingCandidate:
        if not isinstance(candidate, MappingCandidate):
            candidate = MappingCandidate.from_dict(candidate)
        return candidate


def _flatten_features(prefix: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = "{}.{}".format(prefix, key) if prefix else str(key)
            result.update(_flatten_features(name, item))
        return result
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {prefix: value}
    return {}


def candidate_features(candidate: MappingCandidate) -> Dict[str, Any]:
    candidate = SearchableOperator.validate_candidate(candidate)
    features = _flatten_features("strategy", candidate.strategy)
    for prefix, values in (("graph", candidate.graph), ("mapping", candidate.mapping), ("precision", candidate.precision), ("layout", candidate.layout), ("resource", candidate.resource_requirements)):
        features.update(_flatten_features(prefix, values))
    return features


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
    selected = list(ranked_ids) if ranked_ids is not None else []
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
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        unique.append(candidate)
    return unique
