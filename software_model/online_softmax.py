"""Calibrated Online Softmax state-update candidates for XSAI.

This module models the update performed for one score tile. P*V, output
rescaling, and final normalization are modeled by FusedAttention, which invokes
this kernel once per Q/KV tile pair.
"""

import json
import math
import os
from math import ceil

from llmcompass_utils import size
from hardware_model.device import Device
from software_model.design_point_recorder import get_active_recorder
from software_model.operators import Operator
from software_model.search_protocol import (
    MappingCandidate,
    ProviderProtocolError,
    deduplicate_candidates,
    validate_ranked_ids,
)
from software_model.softmax import (
    estimate_analytical_online_state_cycles,
    estimate_analytical_softmax_cycles,
)
from software_model.utils import DataType, Tensor


class OnlineSoftmaxUpdate(Operator):
    """Model one Online Softmax score-tile update with carried row state."""

    CALIBRATION_FILE = "online_softmax_calibration.json"
    _calibration_cache = None

    # Each entry corresponds to an XSAI implementation with a recorded cycle
    # count. The fields are strategy properties, not a speculative Cartesian
    # product of independent optimizations.
    VARIANTS = (
        {
            "name": "rvv_poly_1row",
            "requires_vfexp2": False,
            "row_group": 1,
            "lmul": 8,
            "vector_registers": 16,
            "exp_impl": "rvv_poly",
            "exp_transform": "generic",
            "scale_path": "generic",
            "alpha_impl": "scalar_exp",
            "row_interleave": "one_row",
            "register_scheme": "1k",
        },
        {
            "name": "vfexp2_1row_lmul8",
            "requires_vfexp2": True,
            "row_group": 1,
            "lmul": 8,
            "vector_registers": 16,
            "exp_impl": "vfexp2",
            "exp_transform": "generic",
            "scale_path": "generic",
            "alpha_impl": "scalar_exp",
            "row_interleave": "one_row",
            "register_scheme": "1k",
        },
        {
            "name": "vfexp2_2row_lmul8_2k",
            "requires_vfexp2": True,
            "row_group": 2,
            "lmul": 8,
            "vector_registers": 32,
            "exp_impl": "vfexp2",
            "exp_transform": "generic",
            "scale_path": "generic",
            "alpha_impl": "scalar_exp",
            "row_interleave": "two_row",
            "register_scheme": "2k",
        },
        {
            "name": "vfexp2_4row_lmul4_2k",
            "requires_vfexp2": True,
            "row_group": 4,
            "lmul": 4,
            "vector_registers": 32,
            "exp_impl": "vfexp2",
            "exp_transform": "generic",
            "scale_path": "generic",
            "alpha_impl": "scalar_exp",
            "row_interleave": "four_row",
            "register_scheme": "2k",
        },
        {
            "name": "vfexp2_8row_lmul2_2k",
            "requires_vfexp2": True,
            "row_group": 8,
            "lmul": 2,
            "vector_registers": 32,
            "exp_impl": "vfexp2",
            "exp_transform": "generic",
            "scale_path": "generic",
            "alpha_impl": "scalar_exp",
            "row_interleave": "eight_row",
            "register_scheme": "2k",
        },
        {
            "name": "vfexp2_4row_positive_scale_vector_alpha",
            "requires_vfexp2": True,
            "row_group": 4,
            "lmul": 4,
            "vector_registers": 32,
            "exp_impl": "vfexp2",
            "exp_transform": "affine",
            "scale_path": "positive_fast_path",
            "alpha_impl": "vector_vfexp2",
            "row_interleave": "four_row",
            "register_scheme": "2k",
        },
    )

    _VARIANTS_BY_NAME = {variant["name"]: variant for variant in VARIANTS}

    def __init__(self, data_type: DataType, cost_model="calibrated"):
        super().__init__(0, 0, 0, 0, data_type)
        if cost_model not in ("analytical", "calibrated"):
            raise ValueError("OnlineSoftmax cost_model must be analytical or calibrated")
        self.cost_model = cost_model
        self.shape = None
        self.scale = 1.0
        self.first = False
        self.last_cycle_breakdown = {}

    class ComputationalGraph:
        def __init__(self, M, N, data_type, scale, first):
            self.M = int(M)
            self.N = int(N)
            self.data_type = data_type
            self.scale = float(scale)
            self.first = bool(first)

    class Mapping:
        def __init__(self, l1_tile_M, l1_tile_N, row_group, lmul):
            self.l1_tile_M = int(l1_tile_M)
            self.l1_tile_N = int(l1_tile_N)
            self.row_group = int(row_group)
            self.lmul = int(lmul)

    def __call__(self, input: Tensor, scale=1.0, first=False) -> Tensor:
        if self.data_type != input.data_type:
            raise ValueError("OnlineSoftmax input data type does not match operator")
        if not math.isfinite(float(scale)):
            raise ValueError("OnlineSoftmax scale must be finite")
        self.shape = list(input.shape)
        self.M = size(input.shape[:-1])
        self.N = int(input.shape[-1])
        if self.M < 1 or self.N < 1:
            raise ValueError("OnlineSoftmax requires positive score-tile dimensions")
        self.scale = float(scale)
        self.first = bool(first)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type, self.scale, self.first
        )
        # The XSAI kernel updates scores in place and carries row state outside
        # the tile. Tensor is shape-only in LLMCompass, so returning it models
        # that in-place score-tile contract.
        return input

    @classmethod
    def calibration(cls):
        if cls._calibration_cache is None:
            path = os.path.join(os.path.dirname(__file__), cls.CALIBRATION_FILE)
            with open(path, "r") as stream:
                data = json.load(stream)
            required = {"profile", "reference_hardware", "reference_shape", "variants"}
            if required - set(data):
                raise RuntimeError("OnlineSoftmax calibration file is incomplete")
            cls._calibration_cache = data
        return cls._calibration_cache

    @staticmethod
    def update_state(previous_max, previous_sum, tile_max, tile_sum, first):
        """Return the mathematical state update used by the XSAI tile kernel."""
        current_max = tile_max if first else max(previous_max, tile_max)
        alpha = 0.0 if first else math.exp(previous_max - current_max)
        current_sum = tile_sum if first else alpha * previous_sum + tile_sum
        return current_max, current_sum, alpha

    def _vector_unit(self, pcb_module):
        return pcb_module.compute_module.core.vector_unit

    def _mapping_for(self, variant, active_rows=None):
        # The calibrated microkernels consume a whole score row. l1_tile_M is
        # only the caller's grouping boundary; row_group remains the kernel's
        # live row-interleave count and can have a residual fallback.
        rows = self.M if active_rows is None else int(active_rows)
        if rows < 1:
            raise ValueError("OnlineSoftmax mapping requires positive rows")
        l1_tile_M = min(rows, max(32, variant["row_group"]))
        return self.Mapping(l1_tile_M, self.N, variant["row_group"], variant["lmul"])

    def _variant_is_supported(self, pcb_module, variant, mapping):
        vector_unit = self._vector_unit(pcb_module)
        if variant["requires_vfexp2"] and not bool(
            getattr(vector_unit, "supports_vfexp2", False)
        ):
            return False
        if variant["lmul"] > int(getattr(vector_unit, "max_lmul", 8)):
            return False
        if variant["vector_registers"] > int(
            getattr(vector_unit, "vector_register_count", 32)
        ):
            return False
        if variant["scale_path"] == "positive_fast_path" and not self.scale > 0.0:
            return False
        data_type = self.computational_graph.data_type
        score_tile_bytes = mapping.l1_tile_M * mapping.l1_tile_N * data_type.word_size
        online_state_bytes = mapping.row_group * 3 * 4
        return score_tile_bytes + online_state_bytes <= pcb_module.compute_module.core.SRAM_size

    def _fallback_variant(self, pcb_module, row_count, preferred):
        """Return the largest hardware-valid kernel that exactly fits the tail."""
        compatible = []
        for variant in self.VARIANTS:
            if variant["row_group"] > row_count:
                continue
            if variant["requires_vfexp2"] != preferred["requires_vfexp2"]:
                continue
            mapping = self._mapping_for(variant, active_rows=row_count)
            if self._variant_is_supported(pcb_module, variant, mapping):
                compatible.append(variant)
        if not compatible:
            return None
        calibration = self.calibration()["variants"]
        return min(
            compatible,
            key=lambda item: (
                -item["row_group"],
                float(calibration[item["name"]]["cycles_per_element"]),
                item["name"],
            ),
        )

    def _build_group_plan(self, pcb_module, preferred):
        """Build an exact, hardware-valid row plan without masked-tail assumptions."""
        preferred_mapping = self._mapping_for(preferred)
        if not self._variant_is_supported(
            pcb_module, preferred, preferred_mapping
        ):
            return None

        row_group = int(preferred["row_group"])
        full_groups, remaining = divmod(self.M, row_group)
        if full_groups == 0:
            return None

        plan = [(preferred, full_groups * row_group)]
        while remaining:
            fallback = self._fallback_variant(
                pcb_module, remaining, preferred
            )
            if fallback is None:
                return None
            rows = int(fallback["row_group"])
            plan.append((fallback, rows))
            remaining -= rows
        return plan

    @staticmethod
    def _tail_metadata(plan):
        tail_variants = [variant["name"] for variant, _ in plan[1:]]
        return {
            "tail_policy": "exact_fallback" if tail_variants else "none",
            "tail_kernel_count": len(tail_variants),
        }

    def _candidate(self, pcb_module, variant):
        mapping = self._mapping_for(variant)
        plan = self._build_group_plan(pcb_module, variant)
        if plan is None:
            return None
        data_type = self.computational_graph.data_type
        score_tile_bytes = mapping.l1_tile_M * mapping.l1_tile_N * data_type.word_size
        online_state_bytes = mapping.row_group * 3 * 4
        l1_bytes = score_tile_bytes + online_state_bytes
        max_vector_registers = variant["vector_registers"]
        max_lmul_required = variant["lmul"]
        for fallback, rows in plan[1:]:
            fallback_mapping = self._mapping_for(fallback, active_rows=rows)
            fallback_score_bytes = (
                fallback_mapping.l1_tile_M
                * fallback_mapping.l1_tile_N
                * data_type.word_size
            )
            fallback_state_bytes = fallback_mapping.row_group * 3 * 4
            l1_bytes = max(
                l1_bytes, fallback_score_bytes + fallback_state_bytes
            )
            max_vector_registers = max(
                max_vector_registers, fallback["vector_registers"]
            )
            max_lmul_required = max(max_lmul_required, fallback["lmul"])
        calibration = self.calibration()
        tail_metadata = self._tail_metadata(plan)
        strategy = {
            "algorithm": "online_softmax_update",
            "kernel_variant": variant["name"],
            "softmax": {
                "exp_impl": variant["exp_impl"],
                "exp_transform": variant["exp_transform"],
                "scale_path": variant["scale_path"],
                "reduction": "single_accumulator",
                "alpha_impl": variant["alpha_impl"],
                "score_output": "tile_fp32",
                "row_interleave": variant["row_interleave"],
                "register_scheme": variant["register_scheme"],
                "cost_model": self.cost_model,
                **tail_metadata,
            },
        }
        return MappingCandidate(
            operator_name=self.recording_name or "OnlineSoftmaxUpdate",
            operator_type="OnlineSoftmaxUpdate",
            execution_kind="online_state_update",
            strategy=strategy,
            graph={
                "M": self.M,
                "N": self.N,
                "scale": self.scale,
                "first": self.first,
            },
            mapping=dict(vars(mapping)),
            mapping_object=mapping,
            precision={
                "input_word_size": data_type.word_size,
                "state_data_type": "fp32",
                "accuracy_class": "bounded_ulp",
            },
            layout={"contiguous_last_dim": True, "scores": "row_major"},
            resource_requirements={
                "l1_bytes": l1_bytes,
                "score_tile_bytes": score_tile_bytes,
                "online_state_bytes": online_state_bytes,
                "vector_registers": max_vector_registers,
                "max_lmul_required": max_lmul_required,
                "requires_vfexp2": variant["requires_vfexp2"],
                "calibration_profile": calibration["profile"],
            },
        )

    def resolve_variant_candidate(
        self, pcb_module, variant_name, allow_fallback_only=False
    ):
        """Resolve a preferred kernel, optionally using fallbacks for the whole tile."""
        preferred = self._VARIANTS_BY_NAME.get(variant_name)
        if preferred is None:
            raise ProviderProtocolError(
                "unknown Online Softmax variant {}".format(variant_name)
            )

        candidate = self._candidate(pcb_module, preferred)
        if candidate is not None or not allow_fallback_only:
            return candidate

        # A fused Attention strategy names the kernel used by its normal Q
        # tiles. A smaller residual Q tile may contain no complete preferred
        # row group, in which case that tile is allowed to run entirely with
        # hardware-valid kernels from the same exponential implementation.
        if self.M >= int(preferred["row_group"]):
            return None
        preferred_mapping = self._mapping_for(preferred)
        if not self._variant_is_supported(
            pcb_module, preferred, preferred_mapping
        ):
            return None

        compatible = []
        calibration = self.calibration()["variants"]
        for fallback in self.VARIANTS:
            if fallback["row_group"] > self.M:
                continue
            if fallback["requires_vfexp2"] != preferred["requires_vfexp2"]:
                continue
            fallback_candidate = self._candidate(pcb_module, fallback)
            if fallback_candidate is not None:
                compatible.append((fallback, fallback_candidate))
        if not compatible:
            return None
        return min(
            compatible,
            key=lambda item: (
                -item[0]["row_group"],
                float(calibration[item[0]["name"]]["cycles_per_element"]),
                item[0]["name"],
            ),
        )[1]

    def enumerate_transfer_candidates(
        self,
        pcb_module: Device,
        generation_mode="heuristic-GPU",
        allow_empty=False,
    ):
        if generation_mode not in (
            "heuristic-GPU",
            "heuristic-our-throughput",
            "heuristic-TPU",
            "exhaustive",
        ):
            raise ValueError("unsupported OnlineSoftmax candidate generation mode")
        if not hasattr(self, "computational_graph"):
            raise RuntimeError("call OnlineSoftmaxUpdate before enumerating candidates")
        candidates = [
            candidate
            for variant in self.VARIANTS
            for candidate in (self._candidate(pcb_module, variant),)
            if candidate is not None
        ]
        candidates = deduplicate_candidates(candidates)
        if not candidates and not allow_empty:
            raise ValueError("no OnlineSoftmax candidates are valid for this hardware")
        return candidates

    def _validated_group_plan(self, pcb_module, candidate):
        variant_name = candidate.strategy["kernel_variant"]
        preferred = self._VARIANTS_BY_NAME.get(variant_name)
        if preferred is None:
            raise ProviderProtocolError(
                "unknown Online Softmax variant {}".format(variant_name)
            )
        plan = self._build_group_plan(pcb_module, preferred)
        if plan is None:
            raise ProviderProtocolError(
                "Online Softmax variant {} has no hardware-valid tail plan".format(
                    variant_name
                )
            )
        declared = candidate.strategy.get("softmax", {})
        expected = self._tail_metadata(plan)
        for name, value in expected.items():
            if name in declared and declared[name] != value:
                raise ProviderProtocolError(
                    "Online Softmax candidate {} does not match the current "
                    "hardware tail plan".format(variant_name)
                )
        return preferred, plan

    def _calibrated_cycles(self, pcb_module, candidate):
        preferred, plan = self._validated_group_plan(pcb_module, candidate)
        variant_name = preferred["name"]
        calibration = self.calibration()
        try:
            anchor = calibration["variants"][variant_name]
        except KeyError as exc:
            raise ProviderProtocolError(
                "missing calibration for {}".format(variant_name)
            ) from exc
        elements = self.M * self.N
        reference_hw = calibration["reference_hardware"]
        vector_unit = self._vector_unit(pcb_module)
        weighted_cycles = 0.0
        groups = []
        cycles_per_element = None
        for group_variant, rows in plan:
            group_anchor = calibration["variants"][group_variant["name"]]
            group_cycles_per_element = float(
                group_anchor["cycles_per_element"]
            )
            if group_variant["requires_vfexp2"]:
                actual = int(
                    getattr(vector_unit, "vfexp2_elements_per_cycle", 0)
                )
                if actual < 1:
                    raise ProviderProtocolError(
                        "vfexp2 candidate requires throughput metadata"
                    )
                group_cycles_per_element *= (
                    float(reference_hw["vfexp2_elements_per_cycle"]) / actual
                )
            if group_variant is preferred:
                cycles_per_element = group_cycles_per_element
            weighted_cycles += rows * self.N * group_cycles_per_element
            groups.append((group_variant["name"], rows))
        cycles = max(1, ceil(weighted_cycles))
        reference_shape = calibration["reference_shape"]
        exact_shape = (
            self.M == int(reference_shape["rows"])
            and self.N == int(reference_shape["cols"])
            and self.first == bool(reference_shape["first"])
            and self.scale == float(reference_shape["scale"])
        )
        exact_hardware = (
            getattr(vector_unit, "online_softmax_calibration_profile", None)
            == calibration["profile"]
            and (
                not preferred["requires_vfexp2"]
                or int(getattr(vector_unit, "vfexp2_elements_per_cycle", 0))
                == int(reference_hw["vfexp2_elements_per_cycle"])
                and int(getattr(vector_unit, "vfexp2_latency_cycles", 0))
                == int(reference_hw["vfexp2_latency_cycles"])
            )
        )
        exact_calibration = exact_shape and exact_hardware and len(plan) == 1
        if exact_calibration:
            cycles = int(anchor["cycles"])
        self.last_cycle_breakdown = {
            "model": "calibrated_throughput",
            "kernel_cycles": cycles,
            "elements": elements,
            "cycles_per_element": cycles_per_element,
            "groups": groups,
            "calibration_variant": variant_name,
            "calibration_exact": bool(exact_calibration),
            "calibration_profile": calibration["profile"],
        }
        return cycles

    @classmethod
    def _variant_factor(cls, variant_name):
        """Return a relative factor against the measured scalar software kernel."""
        calibration = cls.calibration()
        try:
            baseline = float(
                calibration["variants"]["rvv_poly_1row"]["cycles_per_element"]
            )
            value = float(calibration["variants"][variant_name]["cycles_per_element"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderProtocolError(
                "missing analytical factor for Online Softmax variant {}".format(
                    variant_name
                )
            ) from exc
        if baseline <= 0 or value <= 0:
            raise ProviderProtocolError("Online Softmax calibration factors must be positive")
        return value / baseline

    def _grouped_variant_cycles(self, per_row_cycles, plan):
        """Apply each validated row-group kernel's calibrated relative factor."""
        cycles = 0.0
        groups = []
        for variant, rows in plan:
            cycles += (
                rows
                * per_row_cycles
                * self._variant_factor(variant["name"])
            )
            groups.append((variant["name"], rows))
        return cycles, groups

    def _analytical_cycles(self, pcb_module, candidate):
        variant, plan = self._validated_group_plan(pcb_module, candidate)
        variant_name = variant["name"]

        # Apply the measured kernel factor to complete row groups. A residual
        # group uses a smaller compatible kernel instead of receiving the full
        # throughput benefit of an 8-row implementation.
        row_group = int(variant["row_group"])
        vector_unit = pcb_module.compute_module.core.vector_unit
        vector_flops_per_cycle = max(
            1, vector_unit.total_vector_flops_per_cycle
        )
        base_per_row = (
            self.N * (vector_unit.flops_per_exp * 3 + 7)
            / vector_flops_per_cycle
        )
        state_per_row = 0.0 if self.first else (
            vector_unit.flops_per_exp + 4
        ) / vector_flops_per_cycle
        selected_factor = self._variant_factor(variant_name)
        weighted_cycles, groups = self._grouped_variant_cycles(
            base_per_row + state_per_row, plan
        )
        cycles = max(1, ceil(weighted_cycles))
        full_rows = self.M // row_group * row_group
        residual_rows = self.M - full_rows
        fallback_name = groups[-1][0] if residual_rows else None

        self.last_cycle_breakdown = {
            "model": "analytical_vector_flops_calibrated_variant",
            "kernel_cycles": cycles,
            "base_softmax_cycles": estimate_analytical_softmax_cycles(
                self.M * self.N, pcb_module
            ),
            "online_state_cycles": estimate_analytical_online_state_cycles(
                self.M, self.first, pcb_module
            ),
            "elements": self.M * self.N,
            "rows": self.M,
            "first": self.first,
            "variant": variant_name,
            "variant_factor": selected_factor,
            "row_group": row_group,
            "full_rows": full_rows,
            "residual_rows": residual_rows,
            "fallback_variant": fallback_name,
            "groups": groups,
            "calibration_exact": False,
            "calibration_profile": self.calibration()["profile"],
        }
        return cycles

    def simulate(self, pcb_module, candidate):
        candidate = self.validate_candidate(candidate)
        if self.cost_model == "analytical":
            return self._analytical_cycles(pcb_module, candidate)
        return self._calibrated_cycles(pcb_module, candidate)

    def evaluate_transfer_candidate(self, pcb_module, candidate, trial_sink=None):
        candidate = self.validate_candidate(candidate)
        cycle_count = self.simulate(pcb_module, candidate)
        latency = cycle_count / pcb_module.compute_module.clock_freq
        recorder = get_active_recorder()
        if recorder is not None and self.recording_name is not None:
            recorder.record_operator_mapping_trial(
                operator_name=self.recording_name,
                operator_type="OnlineSoftmaxUpdate",
                execution_kind=candidate.execution_kind,
                graph=candidate.graph,
                mapping=candidate.mapping,
                cycle_count=cycle_count,
                raw_local_latency_s=latency,
                candidate=candidate,
            )
        if trial_sink is not None:
            trial_sink(candidate.to_dict(), latency, cycle_count, candidate.strategy)
        self.best_mapping = candidate.mapping
        self.best_cycle_count = cycle_count
        self.best_latency = latency
        self.execution_kind = candidate.execution_kind
        self.selected_strategy = candidate.strategy
        self.latency = latency
        return latency

    def _compile_and_simulate_transfer_learn(
        self, pcb_module, provider=None, stage=None, hardware=None, trial_sink=None, operator_name=None
    ):
        if provider is None:
            raise ProviderProtocolError("transfer-learn requires a provider")
        candidates = self.enumerate_transfer_candidates(
            pcb_module, getattr(provider, "generation_mode", "heuristic-GPU")
        )
        records = [candidate.to_dict() for candidate in candidates]
        top_k = getattr(provider, "top_k", 1)
        ranked = provider.rank(
            operator_name or self.recording_name or "OnlineSoftmaxUpdate",
            stage,
            hardware or {},
            {},
            records,
            top_k,
        )
        selected_ids = validate_ranked_ids(records, ranked, top_k)
        candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        best = None
        for candidate_id in selected_ids:
            candidate = candidates_by_id[candidate_id]
            latency = self.evaluate_transfer_candidate(pcb_module, candidate, trial_sink)
            if best is None or (latency, candidate.candidate_id) < (
                best[0],
                best[1].candidate_id,
            ):
                best = latency, candidate
        if best is None:
            raise ProviderProtocolError("provider selected no OnlineSoftmax candidate")
        best_latency, best_candidate = best
        best_cycles = self.simulate(pcb_module, best_candidate)
        self.best_mapping = best_candidate.mapping
        self.best_cycle_count = best_cycles
        self.best_latency = best_latency
        self.execution_kind = best_candidate.execution_kind
        self.selected_strategy = best_candidate.strategy
        self.latency = best_latency
        return best_latency

    def compile_and_simulate(
        self, pcb_module: Device, compile_mode=None, provider=None, stage=None,
        hardware=None, trial_sink=None, operator_name=None,
    ):
        if not hasattr(self, "computational_graph"):
            raise RuntimeError("call OnlineSoftmaxUpdate before compiling")
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        if compile_mode == "transfer-learn":
            return self._compile_and_simulate_transfer_learn(
                pcb_module, provider, stage, hardware, trial_sink, operator_name
            )
        best = None
        for candidate in self.enumerate_transfer_candidates(pcb_module, compile_mode or "heuristic-GPU"):
            latency = self.evaluate_transfer_candidate(pcb_module, candidate, trial_sink)
            if best is None or (latency, candidate.candidate_id) < (
                best[0],
                best[1].candidate_id,
            ):
                best = latency, candidate
        if best is None:
            raise ValueError("no OnlineSoftmax candidate was evaluated")
        best_latency, best_candidate = best
        best_cycles = self.simulate(pcb_module, best_candidate)
        self.best_mapping = best_candidate.mapping
        self.best_cycle_count = best_cycles
        self.best_latency = best_latency
        self.execution_kind = best_candidate.execution_kind
        self.selected_strategy = best_candidate.strategy
        self.latency = best_latency
        return best_latency

    def roofline_model(self, pcb_module: Device):
        candidates = self.enumerate_transfer_candidates(pcb_module)
        cycle_counts = [self.simulate(pcb_module, candidate) for candidate in candidates]
        self.roofline_latency = min(cycle_counts) / pcb_module.compute_module.clock_freq
        return self.roofline_latency
