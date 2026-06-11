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
    return {field: getattr(mapping, field, None) for field in fields}


def _graph_values(operator, fields) -> Dict[str, Any]:
    graph = getattr(operator, "computational_graph", operator)
    return {field: getattr(graph, field) for field in fields}


def _matmul_result(operator, mapping_source: str) -> Dict[str, Any]:
    result = {
        "mapping_source": mapping_source,
        "execution_kind": operator.execution_kind,
        "strategy": getattr(operator, "selected_strategy", None),
        **_graph_values(operator, ("M", "N", "K")),
        **_mapping_values(operator.best_mapping, MATMUL_MAPPING_FIELDS),
        "local_latency_s": operator.best_latency,
    }
    return result


def extract_best_software_mapping(model) -> Dict[str, Dict[str, Any]]:
    result = {}
    for result_name, attribute_name, mapping_source in MATMUL_MODELS:
        result[result_name] = _matmul_result(
            getattr(model, attribute_name), mapping_source
        )

    softmax = model.A_softmax
    result["A_softmax"] = {
        "mapping_source": "searched",
        **_graph_values(softmax, ("M", "N")),
        **_mapping_values(softmax.best_mapping, SOFTMAX_MAPPING_FIELDS),
        "local_latency_s": softmax.best_latency,
    }

    layer_norm = model.layer_norm0
    result["layer_norm"] = {
        "mapping_source": "layer_norm0_reused",
        **_graph_values(layer_norm, ("M", "N")),
        **_mapping_values(layer_norm.best_mapping, LAYERNORM_MAPPING_FIELDS),
        "local_latency_s": layer_norm.best_latency,
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
