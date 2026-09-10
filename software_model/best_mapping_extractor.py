from typing import Any, Dict

from software_model.design_point_recorder import MAIN_RESULT_COLUMNS


MATMUL_MAPPING_FIELDS = (
    "l2_tile_M",
    "l2_tile_N",
    "l2_tile_K",
    "is_l2_double_buffering",
    "l1_tile_M",
    "l1_tile_N",
    "l1_tile_K",
    "l2_loop_order",
    "l1_loop_order",
    "l0_M_tiling_factor",
    "l0_N_tiling_factor",
    "l0_K_tiling_factor",
    "dataflow",
)
SOFTMAX_MAPPING_FIELDS = (
    "l2_tile_M",
    "l2_tile_N",
    "is_l2_double_buffering",
    "l1_tile_M",
    "l1_tile_N",
    "is_l1_double_buffering",
)
LAYERNORM_MAPPING_FIELDS = ("l2_tile_M", "l2_tile_N", "l1_tile_M", "l1_tile_N")

MATMUL_MODELS = (
    ("QKV_proj", "Q_proj", "Q_proj_reused_for_qkv"),
    ("Q_mul_K", "Q_mul_K", "searched"),
    ("A_mul_V", "A_mul_V", "searched"),
    ("H_matmul0", "H_matmul0", "searched"),
    ("H_matmul1", "H_matmul1", "searched"),
    ("H_matmul2", "H_matmul2", "searched"),
)


def _mapping_values(mapping: Any, fields) -> Dict[str, Any]:
    if mapping is None:
        return {field: None for field in fields}
    if isinstance(mapping, dict):
        return {field: mapping.get(field) for field in fields}
    return {field: getattr(mapping, field, None) for field in fields}


def _graph_values(operator, fields) -> Dict[str, Any]:
    graph = getattr(operator, "computational_graph", operator)
    return {field: getattr(graph, field) for field in fields}


def _matmul_result(operator, mapping_source: str, latency=None, strategy=None) -> Dict[str, Any]:
    result = {
        "mapping_source": mapping_source,
        "execution_kind": getattr(operator, "execution_kind", "uncompiled"),
        "strategy": getattr(operator, "selected_strategy", None) if strategy is None else strategy,
        **_graph_values(operator, ("M", "N", "K")),
        **_mapping_values(getattr(operator, "best_mapping", None), MATMUL_MAPPING_FIELDS),
        "local_latency_s": getattr(operator, "best_latency", None) if latency is None else latency,
    }
    return result


def extract_best_software_mapping(model) -> Dict[str, Dict[str, Any]]:
    result = {}
    for result_name, attribute_name, mapping_source in MATMUL_MODELS:
        result[result_name] = _matmul_result(
            getattr(model, attribute_name), mapping_source
        )

    attention = getattr(model, "attention", None)
    attention_breakdown = getattr(attention, "last_latency_breakdown", {}) if attention is not None else {}
    attention_latency = getattr(attention, "best_latency", None) if attention is not None else None
    attention_strategy = getattr(attention, "selected_strategy", None) if attention is not None else None
    if attention is not None and hasattr(attention, "M"):
        result["attention"] = {
            "mapping_source": "searched",
            "execution_kind": getattr(attention, "execution_kind", "uncompiled"),
            "strategy": attention_strategy,
            "batch_heads": getattr(attention, "batch_heads", None),
            "M": attention.M,
            "N": attention.N,
            "K": attention.K,
            "mapping": getattr(attention, "best_mapping", None),
            "local_latency_s": attention_latency,
            "breakdown": dict(attention_breakdown),
        }
        if "qk" in attention_breakdown and "Q_mul_K" in result:
            result["Q_mul_K"]["mapping_source"] = "attention_fused"
            result["Q_mul_K"]["execution_kind"] = "attention_subgraph"
            result["Q_mul_K"]["strategy"] = attention_strategy
            result["Q_mul_K"]["local_latency_s"] = attention_breakdown["qk"]
        if "pv" in attention_breakdown and "A_mul_V" in result:
            result["A_mul_V"]["mapping_source"] = "attention_fused"
            result["A_mul_V"]["execution_kind"] = "attention_subgraph"
            result["A_mul_V"]["strategy"] = attention_strategy
            result["A_mul_V"]["local_latency_s"] = attention_breakdown["pv"]

    softmax = model.A_softmax
    result["A_softmax"] = {
        "mapping_source": "searched",
        **_graph_values(softmax, ("M", "N")),
        **_mapping_values(getattr(softmax, "best_mapping", None), SOFTMAX_MAPPING_FIELDS),
        "local_latency_s": getattr(softmax, "best_latency", None),
    }
    if "softmax" in attention_breakdown:
        result["A_softmax"]["mapping_source"] = "attention_fused"
        result["A_softmax"]["local_latency_s"] = attention_breakdown["softmax"]

    layer_norm = model.layer_norm0
    result["layer_norm"] = {
        "mapping_source": "layer_norm0_reused",
        **_graph_values(layer_norm, ("M", "N")),
        **_mapping_values(layer_norm.best_mapping, LAYERNORM_MAPPING_FIELDS),
        "local_latency_s": layer_norm.best_latency,
    }

    gelu = model.H_gelu
    result["H_gelu"] = {
        "mapping_source": "searched",
        "execution_kind": gelu.execution_kind,
        **_graph_values(gelu, ("M",)),
        **_mapping_values(gelu.best_mapping, ("vector_factor",)),
        "local_latency_s": gelu.best_latency,
    }
    return result


def flatten_best_software_mapping(model) -> Dict[str, Any]:
    extracted = extract_best_software_mapping(model)
    flattened = {
        f"{operator}_{field}": value
        for operator, values in extracted.items()
        for field, value in values.items()
    }
    return {column: flattened.get(column) for column, _ in MAIN_RESULT_COLUMNS}
