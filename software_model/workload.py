"""Strict, framework-independent model/workload configuration contract."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

# Bump when latency/memory estimates or operator semantics change.
# Operator caches and trial identities rely on this manual version.
# Backend result-cache invalidation is controlled separately by CACHE_VERSION.
EVALUATOR_VERSION = "transformer-layers-v2"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fields_only(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a dictionary")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")


def integer(value, path, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class LayerSpec:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    norm_type: str
    ffn_type: str
    position_embedding: str
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    qk_norm: bool = False
    attention_bias: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden_size: int
    layers: tuple
    vocab_size: int = None
    tie_word_embeddings: bool = False
    max_position_embeddings: int = None

    @property
    def num_layers(self):
        return len(self.layers)

    def to_dict(self):
        return {"name": self.name, "hidden_size": self.hidden_size, "num_layers": self.num_layers,
                "layers": [x.to_dict() for x in self.layers], "vocab_size": self.vocab_size,
                "tie_word_embeddings": self.tie_word_embeddings,
                "max_position_embeddings": self.max_position_embeddings}


@dataclass(frozen=True)
class WorkloadSpec:
    batch_size: int
    input_seq_len: int
    decode_context_len: int
    dtype: str

    def to_dict(self):
        return asdict(self)


def _layer(value, hidden, path):
    allowed = set(LayerSpec.__dataclass_fields__) - {"hidden_size"}
    fields_only(value, allowed, path)
    required = allowed - {"norm_eps", "rope_theta", "qk_norm", "attention_bias"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    layer = LayerSpec(hidden_size=hidden, **value)
    for field in ("num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size"):
        integer(getattr(layer, field), f"{path}.{field}")
    if layer.num_attention_heads % layer.num_key_value_heads:
        raise ValueError(f"{path}.num_key_value_heads must divide num_attention_heads")
    for field, supported in (("norm_type", ("layernorm", "rmsnorm")), ("ffn_type", ("gelu", "swiglu")),
                             ("position_embedding", ("learned_absolute", "rope", "none"))):
        if getattr(layer, field) not in supported:
            raise ValueError(f"{path}.{field}: supported values are {supported}")
    for field in ("qk_norm", "attention_bias"):
        if not isinstance(getattr(layer, field), bool):
            raise ValueError(f"{path}.{field} must be boolean")
    for field in ("norm_eps", "rope_theta"):
        value = getattr(layer, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}.{field} must be finite and positive")
    return replace(layer, norm_eps=float(layer.norm_eps), rope_theta=float(layer.rope_theta))


def parse_workload(config):
    fields_only(config, ("schema_version", "model", "workload"), "config")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    model = config.get("model")
    fields_only(model, ("name", "num_layers", "hidden_size", "layer_defaults", "layer_overrides",
                        "vocab_size", "tie_word_embeddings", "max_position_embeddings"), "model")
    n = integer(model.get("num_layers"), "model.num_layers")
    hidden = integer(model.get("hidden_size"), "model.hidden_size")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ValueError("model.name must be a nonempty string")
    defaults = model.get("layer_defaults")
    _layer(defaults, hidden, "model.layer_defaults")
    layers = [dict(defaults) for _ in range(n)]
    overrides = model.get("layer_overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("model.layer_overrides must be a list")
    seen = set()
    for override in overrides:
        fields_only(override, (set(LayerSpec.__dataclass_fields__) - {"hidden_size"}) | {"indices"}, "layer_overrides")
        indices = override.get("indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("layer_overrides.indices must be a nonempty list")
        for i in indices:
            integer(i, "layer_overrides.indices", 0)
            if i >= n or i in seen:
                raise ValueError(f"layer {i}: override index out of range or repeated")
            seen.add(i)
            layers[i].update({k: v for k, v in override.items() if k != "indices"})
    for field in ("vocab_size", "max_position_embeddings"):
        if model.get(field) is not None:
            integer(model[field], "model." + field)
    if not isinstance(model.get("tie_word_embeddings", False), bool):
        raise ValueError("model.tie_word_embeddings must be boolean")
    spec = ModelSpec(model["name"], hidden, tuple(_layer(x, hidden, f"layer {i}") for i, x in enumerate(layers)),
                     model.get("vocab_size"), model.get("tie_word_embeddings", False), model.get("max_position_embeddings"))
    w = config.get("workload")
    fields_only(w, WorkloadSpec.__dataclass_fields__, "workload")
    if set(w) != set(WorkloadSpec.__dataclass_fields__):
        raise ValueError("workload requires batch_size, input_seq_len, decode_context_len and dtype")
    workload = WorkloadSpec(integer(w["batch_size"], "batch_size"), integer(w["input_seq_len"], "input_seq_len"),
                            integer(w["decode_context_len"], "decode_context_len", 0), w["dtype"])
    if workload.dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError("workload.dtype must be fp16, bf16 or fp32; quantization is unsupported")
    if spec.max_position_embeddings is not None and max(workload.input_seq_len, workload.decode_context_len + 1) > spec.max_position_embeddings:
        raise ValueError("workload exceeds model.max_position_embeddings")
    return spec, workload


def normalized_workload(model, workload):
    return {"schema_version": 1, "model": model.to_dict(), "workload": workload.to_dict()}


def legacy_task_config(task):
    t = {k.replace("-", "_"): v for k, v in task.items()}
    fields_only(t, ("batch_size", "d_model", "n_heads", "input_seq_len", "data_type", "num_layers", "decode_context_len"), "task")
    if "num_layers" not in t:
        raise ValueError("legacy task requires explicit num-layers; migrate to workload-config")
    h, nq = integer(t.get("d_model", 12288), "d_model"), integer(t.get("n_heads", 96), "n_heads")
    if h % nq:
        raise ValueError("legacy task n_heads must divide d_model")
    return {"schema_version": 1, "model": {"name": "legacy-task", "num_layers": t["num_layers"], "hidden_size": h,
            "layer_defaults": {"num_attention_heads": nq, "num_key_value_heads": nq, "head_dim": h // nq,
            "intermediate_size": 4*h, "norm_type": "layernorm", "ffn_type": "gelu", "position_embedding": "learned_absolute"}},
            "workload": {"batch_size": t.get("batch_size", 1), "input_seq_len": t.get("input_seq_len", 2048),
            "decode_context_len": t.get("decode_context_len", t.get("input_seq_len", 2048)), "dtype": t.get("data_type", "fp16")}}
