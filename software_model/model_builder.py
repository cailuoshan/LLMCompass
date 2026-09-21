"""Graph factory, projection combinations and workload feasibility."""
import copy
import itertools
import math
from software_model.transformer import TransformerModel
from software_model.workload import integer


class InfeasibleWorkload(ValueError):
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("infeasible workload: " + "; ".join(self.reasons))


def projection_configs(model, projection_options=None):
    """Enumerate combinations of separate/fused projection implementations."""
    space = {k.replace("-", "_"):v for k,v in (projection_options or {}).items()}
    if set(space) - {"qkv_projection", "gate_up_projection"}:
        raise ValueError("unknown projection-options field; fused Attention is not implemented")
    values = {}
    for key in ("qkv_projection", "gate_up_projection"):
        choices = space.get(key, ["separate"])
        if not isinstance(choices,list) or not choices or any(x not in ("separate","fused") for x in choices):
            raise ValueError(f"{key} requires a nonempty list of separate/fused")
        values[key] = sorted(set(choices))
    if all(x.ffn_type == "gelu" for x in model.layers):
        values["gate_up_projection"] = ["separate"]
    return [dict(zip(values, items)) for items in itertools.product(*values.values())]


def build_model_graph(model_spec, workload_spec, stage, device_count, software_config=None):
    integer(device_count, "device_count")
    return TransformerModel(model_spec,workload_spec,stage,device_count,software_config)


def precision_specs(hardware, dtype):
    """Use explicit template capabilities, never infer BF16 from byte width."""
    specs = copy.deepcopy(hardware)
    core = specs["device"]["compute_chiplet"]["core"]
    matrix, vector = core["systolic_array"], core["vector_unit"]
    if matrix["data_type"] == dtype and vector["data_type"] == dtype:
        return specs
    mode = core.get("precision_modes", {}).get(dtype)
    if not isinstance(mode,dict):
        raise ValueError(f"hardware has no explicit {dtype} matrix/vector precision mode")
    for key in ("matrix_mac_per_cycle", "vector_flop_per_cycle"):
        if isinstance(mode.get(key),bool) or not isinstance(mode.get(key),(int,float)) or not math.isfinite(mode[key]) or mode[key] <= 0:
            raise ValueError(f"precision_modes.{dtype}.{key} must be positive")
    if mode.get("accumulation_dtype") != "fp32":
        raise ValueError("configured floating point mode requires explicit fp32 accumulation")
    matrix.update(data_type=dtype,mac_per_cycle=mode["matrix_mac_per_cycle"])
    vector.update(data_type=dtype,flop_per_cycle=mode["vector_flop_per_cycle"])
    return specs


def check_feasibility(model, workload, hardware, projection_options=None):
    reasons = []
    p = hardware["device_count"]
    try:
        integer(p, "device_count")
        precision_specs(hardware, workload.dtype)
    except ValueError as exc:
        reasons.append(str(exc))
    if isinstance(p,int) and not isinstance(p,bool) and p > 0:
        for i,layer in enumerate(model.layers):
            for key in ("num_attention_heads","num_key_value_heads","intermediate_size"):
                if getattr(layer,key) % p:
                    reasons.append(f"layer {i}: TP={p} must divide {key}={getattr(layer,key)}")
    policies = projection_configs(model,projection_options)
    valid, memory = [], {}
    if not reasons:
        capacity = hardware["device"]["memory"]["total_capacity_GB"] * 1024**3
        for policy in policies:
            summaries = {stage:build_model_graph(model,workload,stage,p,policy).memory_summary() for stage in ("prefill","decode")}
            if max(x["peak_bytes"] for x in summaries.values()) <= capacity:
                valid.append(policy)
                memory[str(policy)] = summaries
        if not valid:
            reasons.append("all software policies exceed per-device memory capacity for Transformer layers")
    return {"feasible":not reasons, "reasons":reasons, "software_configs":valid, "memory":memory}
