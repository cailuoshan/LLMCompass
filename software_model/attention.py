"""Searchable materialized, online, and FlashAttention implementations."""

import heapq
import math
from array import array
from math import ceil

from llmcompass_utils import size
from hardware_model.device import Device
from software_model.design_point_recorder import get_active_recorder
from software_model.matmul import BatchedMatmul
from software_model.online_softmax import OnlineSoftmaxUpdate
from software_model.operators import Operator
from software_model.search_protocol import (
    MappingCandidate,
    ProviderProtocolError,
    deduplicate_candidates,
    validate_ranked_ids,
)
from software_model.softmax import (
    Softmax,
    estimate_analytical_online_state_cycles,
    estimate_analytical_softmax_cycles,
)
from software_model.utils import DataType, Tensor


class FusedAttention(Operator):
    """Model QK, softmax, and PV as one searchable attention subgraph.

    The materialized candidate delegates to the original three operators. The
    online and flash candidates use an analytical tiled dataflow model, while
    reusing OnlineSoftmaxUpdate for the common Softmax and row-state costs.
    """

    SOFTMAX_COST_MODEL = "analytical"

    class Mapping:
        def __init__(
            self,
            q_tile_M,
            kv_tile_N,
            head_dim_tile_K,
            row_group,
            lmul,
            is_kv_double_buffering,
            storage_policy="l1_resident",
            l1_q_tile_M=None,
            l1_kv_tile_N=None,
        ):
            self.q_tile_M = int(q_tile_M)
            self.kv_tile_N = int(kv_tile_N)
            self.head_dim_tile_K = int(head_dim_tile_K)
            self.row_group = int(row_group)
            self.lmul = int(lmul)
            self.is_kv_double_buffering = bool(is_kv_double_buffering)
            if storage_policy not in {"l1_resident", "l2_staged"}:
                raise ValueError(
                    "unsupported attention storage policy: {}".format(
                        storage_policy
                    )
                )
            self.storage_policy = storage_policy
            self.l1_q_tile_M = int(l1_q_tile_M or q_tile_M)
            self.l1_kv_tile_N = int(l1_kv_tile_N or kv_tile_N)

    def __init__(
        self,
        data_type: DataType,
        qk_matmul=None,
        softmax=None,
        pv_matmul=None,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.qk_matmul = qk_matmul or BatchedMatmul(data_type)
        self.softmax = softmax or Softmax(data_type)
        self.pv_matmul = pv_matmul or BatchedMatmul(data_type)
        # Keep standalone Attention transfer searches operator-name stable.
        for child, name in (
            (self.qk_matmul, "Q_mul_K"),
            (self.softmax, "A_softmax"),
            (self.pv_matmul, "A_mul_V"),
        ):
            if getattr(child, "recording_name", None) is None:
                child.recording_name = name
        self.scale = 1.0
        self.last_cycle_breakdown = {}
        self.last_latency_breakdown = {}
        self._online_kernel_candidate_cache = {}
        self._online_tile_candidate_cache = {}

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, scale=None):
        if not (
            query.data_type == key.data_type == value.data_type == self.data_type
        ):
            raise ValueError("Attention input data types must match")
        if len(query.shape) < 2 or len(key.shape) < 2 or len(value.shape) < 2:
            raise ValueError("Attention inputs must have at least two dimensions")
        if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
            raise ValueError("Attention batch and head dimensions must match")
        self.batch_heads = size(query.shape[:-2])
        self.M = int(query.shape[-2])
        self.K = int(query.shape[-1])
        self.N = int(key.shape[-1])
        if key.shape[-2] != self.K:
            raise ValueError("Attention key head dimension does not match query")
        if value.shape[-2] != self.N or value.shape[-1] != self.K:
            raise ValueError("Attention value shape does not match key/query")
        if min(self.batch_heads, self.M, self.N, self.K) < 1:
            raise ValueError("Attention dimensions must be positive")
        self.scale = float(scale if scale is not None else 1.0 / math.sqrt(self.K))
        if not math.isfinite(self.scale):
            raise ValueError("Attention scale must be finite")

        # Build the legacy subgraph as well. These shape-only calls allow the
        # materialized candidate to use the original simulators unchanged.
        scores = self.qk_matmul(query, key)
        probabilities = self.softmax(scores)
        output = self.pv_matmul(probabilities, value)
        self.output_shape = list(output.shape)
        return output

    def _graph(self):
        return {
            "batch_heads": self.batch_heads,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "scale": self.scale,
        }

    def _materialized_candidate(self):
        score_bytes = (
            self.batch_heads * self.M * self.N * self.data_type.word_size
        )
        return MappingCandidate(
            operator_name=self.recording_name or "Attention",
            operator_type="FusedAttention",
            execution_kind="attention_subgraph",
            strategy={
                "algorithm": "materialized_attention",
                "fusion": "none",
                "softmax": {
                    "score_output": "materialized_native",
                    "reduction": "single_accumulator",
                    "cost_model": self.SOFTMAX_COST_MODEL,
                },
            },
            graph=self._graph(),
            mapping={},
            precision={
                "input_word_size": self.data_type.word_size,
                "score_data_type": "native",
                "accumulator": "native",
                "accuracy_class": "exact",
            },
            layout={"scores": "materialized", "contiguous_last_dim": True},
            resource_requirements={
                "materialized_score_bytes": score_bytes,
                "materialized_probability_bytes": score_bytes,
            },
        )

    def _online_kernel_candidates(self, pcb_module, q_tile, kv_tile):
        cache_key = (id(pcb_module), int(q_tile), int(kv_tile))
        cached = self._online_kernel_candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        operator = OnlineSoftmaxUpdate(
            self.data_type, cost_model=self.SOFTMAX_COST_MODEL
        )
        operator(
            Tensor([q_tile, kv_tile], self.data_type),
            scale=self.scale,
            first=False,
        )
        candidates = operator.enumerate_transfer_candidates(
            pcb_module, "exhaustive", allow_empty=True
        )
        self._online_kernel_candidate_cache[cache_key] = candidates
        return candidates

    def _online_tile_candidates(
        self, pcb_module, q_tile, kv_tile, variant_name
    ):
        """Return the selected variant for every actual residual tile shape."""
        cache_key = (
            id(pcb_module), int(q_tile), int(kv_tile), str(variant_name)
        )
        if cache_key in self._online_tile_candidate_cache:
            return self._online_tile_candidate_cache[cache_key]
        q_rows = {
            min(q_tile, self.M - index * q_tile)
            for index in range(ceil(self.M / q_tile))
        }
        kv_cols = {
            min(kv_tile, self.N - index * kv_tile)
            for index in range(ceil(self.N / kv_tile))
        }
        selected = []
        for rows in sorted(q_rows):
            for cols in sorted(kv_cols):
                operator = OnlineSoftmaxUpdate(
                    self.data_type, cost_model=self.SOFTMAX_COST_MODEL
                )
                operator(
                    Tensor([rows, cols], self.data_type),
                    scale=self.scale,
                    first=False,
                )
                candidate = operator.resolve_variant_candidate(
                    pcb_module,
                    variant_name,
                    allow_fallback_only=True,
                )
                if candidate is None:
                    self._online_tile_candidate_cache[cache_key] = None
                    return None
                selected.append(candidate)
        self._online_tile_candidate_cache[cache_key] = selected
        return selected

    def _working_set(self, mapping, flash):
        word_size = self.data_type.word_size
        q_bytes = mapping.q_tile_M * self.K * word_size
        k_bytes = mapping.kv_tile_N * self.K * word_size
        v_bytes = mapping.kv_tile_N * self.K * word_size
        logical_kv_bytes = k_bytes + v_bytes
        if mapping.is_kv_double_buffering:
            k_bytes *= 2
            v_bytes *= 2
        score_bytes = mapping.q_tile_M * mapping.kv_tile_N * 4
        output_bytes = mapping.q_tile_M * self.K * 4
        state_bytes = mapping.q_tile_M * 2 * 4
        # FlashAttention keeps one score tile live; online_attention also keeps
        # a probability tile because QK/softmax and PV are separate kernels.
        probability_bytes = 0 if flash else score_bytes
        resident_sram_bytes = (
            q_bytes
            + k_bytes
            + v_bytes
            + score_bytes
            + probability_bytes
            + output_bytes
            + state_bytes
        )

        # The L2-staged path executes these smaller L1 stripes. Capacity is
        # based on the same physical shapes used by the event DAG.
        stripe_rows = min(mapping.q_tile_M, mapping.l1_q_tile_M)
        stripe_cols = min(mapping.kv_tile_N, mapping.l1_kv_tile_N)
        stripe_q_bytes = stripe_rows * self.K * word_size
        stripe_score_bytes = stripe_rows * stripe_cols * 4
        stripe_output_bytes = stripe_rows * self.K * 4
        stripe_state_bytes = stripe_rows * 2 * 4
        stream_kv_bytes = 2 * stripe_cols * self.K * word_size
        if mapping.is_kv_double_buffering:
            stream_kv_bytes *= 2
        staged_sram_bytes = (
            stripe_q_bytes
            + stream_kv_bytes
            + stripe_score_bytes
            + (0 if flash else stripe_score_bytes)
            + stripe_output_bytes
            + stripe_state_bytes
        )

        logical_per_head = (
            q_bytes
            + logical_kv_bytes
            + probability_bytes
            + output_bytes
            + state_bytes
        )
        # One L2 slot owns one physical Q stripe, one logical KV tile and its
        # running output/softmax state. Concurrent slots are capacity-limited
        # by the scheduler instead of filtering against every active head.
        l2_slot_bytes = (
            stripe_q_bytes
            + logical_kv_bytes
            + (0 if flash else stripe_score_bytes)
            + stripe_output_bytes
            + stripe_state_bytes
        )
        return {
            "q_tile_bytes": q_bytes,
            "kv_tile_bytes": k_bytes + v_bytes,
            "score_tile_bytes": score_bytes,
            "probability_tile_bytes": probability_bytes,
            "output_accumulator_bytes": output_bytes,
            "online_state_bytes": state_bytes,
            "resident_sram_bytes": resident_sram_bytes,
            "staged_sram_bytes": staged_sram_bytes,
            "logical_working_set_bytes": logical_per_head,
            "l2_working_set_bytes": l2_slot_bytes,
            "l2_slot_bytes": l2_slot_bytes,
            "l1_stripe_rows": stripe_rows,
            "l1_stripe_cols": stripe_cols,
            "q_reload_bytes_per_kv_tile": stripe_q_bytes,
            "state_spill_bytes_per_transition": (
                stripe_output_bytes + stripe_state_bytes
            ),
        }

    @staticmethod
    def _tile_segments(total, logical_tile, l1_tile):
        """Split every logical tile into physical L1 stripes."""
        segments = []
        for outer_index, outer_start in enumerate(range(0, total, logical_tile)):
            outer_size = min(logical_tile, total - outer_start)
            for inner_index, inner_start in enumerate(range(0, outer_size, l1_tile)):
                segments.append(
                    (
                        outer_index,
                        inner_index,
                        outer_start + inner_start,
                        min(l1_tile, outer_size - inner_start),
                    )
                )
        return segments

    def _select_l1_staging_tile(
        self,
        pcb_module,
        mapping,
        flash,
        variant_name,
    ):
        """Choose the largest legal physical stripe that fits local SRAM."""
        q_options = {
            min(mapping.q_tile_M, value)
            for value in (1, 2, 4, 8, 16, 32, 64, 128)
        }
        q_options.add(mapping.q_tile_M)
        q_options.add(min(mapping.q_tile_M, max(1, mapping.row_group)))
        kv_options = {
            min(mapping.kv_tile_N, value)
            for value in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        }
        kv_options.add(mapping.kv_tile_N)
        shapes = sorted(
            {(q_rows, kv_cols) for q_rows in q_options for kv_cols in kv_options},
            key=lambda item: (item[0] * item[1], item[0], item[1]),
            reverse=True,
        )
        sram_size = int(pcb_module.compute_module.core.SRAM_size)
        for q_rows, kv_cols in shapes:
            mapping.l1_q_tile_M = q_rows
            mapping.l1_kv_tile_N = kv_cols
            tile_candidates = self._online_tile_candidates(
                pcb_module, q_rows, kv_cols, variant_name
            )
            if tile_candidates is None:
                continue
            resources = self._working_set(mapping, flash)
            online_l1 = max(
                item.resource_requirements["l1_bytes"]
                for item in tile_candidates
            )
            if max(resources["staged_sram_bytes"], online_l1) <= sram_size:
                return resources, tile_candidates
        return None, None

    def _fused_candidate(
        self,
        pcb_module,
        algorithm,
        mapping,
        online_candidate,
        tile_candidates,
    ):
        flash = algorithm == "flash_attention"
        resources = self._working_set(mapping, flash)
        softmax_strategy = dict(online_candidate.strategy["softmax"])
        tail_counts = [
            int(item.strategy["softmax"]["tail_kernel_count"])
            + int(
                item.strategy["kernel_variant"]
                != online_candidate.strategy["kernel_variant"]
            )
            for item in tile_candidates
        ]
        softmax_strategy["tail_policy"] = (
            "exact_fallback" if any(tail_counts) else "none"
        )
        softmax_strategy["tail_kernel_count"] = max(tail_counts, default=0)
        resources["vector_registers"] = max(
            item.resource_requirements["vector_registers"]
            for item in tile_candidates
        )
        resources["max_lmul_required"] = max(
            item.resource_requirements["max_lmul_required"]
            for item in tile_candidates
        )
        resources["online_softmax_l1_bytes"] = max(
            item.resource_requirements["l1_bytes"]
            for item in tile_candidates
        )
        resident_sram_bytes = max(
            resources["resident_sram_bytes"],
            resources["online_softmax_l1_bytes"],
        )
        sram_size = int(pcb_module.compute_module.core.SRAM_size)
        l2_size = int(pcb_module.compute_module.l2_size)
        if resident_sram_bytes <= sram_size:
            mapping.storage_policy = "l1_resident"
            mapping.l1_q_tile_M = mapping.q_tile_M
            mapping.l1_kv_tile_N = mapping.kv_tile_N
            resources["sram_bytes"] = resident_sram_bytes
        else:
            mapping.storage_policy = "l2_staged"
            resources, tile_candidates = self._select_l1_staging_tile(
                pcb_module,
                mapping,
                flash,
                online_candidate.strategy["kernel_variant"],
            )
            if resources is None:
                return None
            resources["vector_registers"] = max(
                item.resource_requirements["vector_registers"]
                for item in tile_candidates
            )
            resources["max_lmul_required"] = max(
                item.resource_requirements["max_lmul_required"]
                for item in tile_candidates
            )
            resources["online_softmax_l1_bytes"] = max(
                item.resource_requirements["l1_bytes"]
                for item in tile_candidates
            )
            resources["sram_bytes"] = max(
                resources["staged_sram_bytes"],
                resources["online_softmax_l1_bytes"],
            )
            slot_bytes = max(1, int(resources["l2_slot_bytes"]))
            slot_count = l2_size // slot_bytes
            if slot_count < 1:
                return None
            q_stripe_count = len(
                self._tile_segments(
                    self.M, mapping.q_tile_M, mapping.l1_q_tile_M
                )
            )
            resources["l2_slot_count"] = min(
                self.batch_heads * q_stripe_count,
                max(1, int(pcb_module.compute_module.core_count)),
                slot_count,
            )
        if mapping.storage_policy == "l1_resident":
            resources["l2_slot_count"] = 0

        # Staging may introduce additional physical tail shapes. Tail-policy
        # metadata must describe the kernels actually executed by the DAG.
        tail_counts = [
            int(item.strategy["softmax"]["tail_kernel_count"])
            + int(
                item.strategy["kernel_variant"]
                != online_candidate.strategy["kernel_variant"]
            )
            for item in tile_candidates
        ]
        softmax_strategy["tail_policy"] = (
            "exact_fallback" if any(tail_counts) else "none"
        )
        softmax_strategy["tail_kernel_count"] = max(tail_counts, default=0)
        strategy = {
            "algorithm": algorithm,
            "kernel_variant": online_candidate.strategy["kernel_variant"],
            "fusion": "qk_softmax_pv" if flash else "qk_softmax",
            "softmax": softmax_strategy,
        }
        return MappingCandidate(
            operator_name=self.recording_name or "Attention",
            operator_type="FusedAttention",
            execution_kind="tiled_attention",
            strategy=strategy,
            graph=self._graph(),
            mapping=dict(vars(mapping)),
            mapping_object=mapping,
            precision={
                "input_word_size": self.data_type.word_size,
                "score_data_type": "fp32",
                "online_state_data_type": "fp32",
                "accumulator": "fp32",
                "accuracy_class": "bounded_ulp",
            },
            layout={
                "scores": "tile_only",
                "q": "row_major",
                "k": "transposed_last_two_dims",
                "v": "row_major",
            },
            resource_requirements=resources,
        )

    def enumerate_transfer_candidates(self, pcb_module: Device, generation_mode="heuristic-GPU"):
        if generation_mode not in (
            "heuristic-GPU",
            "heuristic-our-throughput",
            "heuristic-TPU",
            "exhaustive",
        ):
            raise ValueError("unsupported Attention candidate generation mode")
        if not hasattr(self, "batch_heads"):
            raise RuntimeError("call FusedAttention before enumerating candidates")
        candidates = [self._materialized_candidate()]
        q_tiles = sorted({min(self.M, value) for value in (64, 128)})
        kv_tiles = sorted({min(self.N, value) for value in (64, 128, 256)})
        head_tile = self.K
        for q_tile in q_tiles:
            for kv_tile in kv_tiles:
                for online_candidate in self._online_kernel_candidates(
                    pcb_module, q_tile, kv_tile
                ):
                    tile_candidates = self._online_tile_candidates(
                        pcb_module,
                        q_tile,
                        kv_tile,
                        online_candidate.strategy["kernel_variant"],
                    )
                    if tile_candidates is None:
                        continue
                    for algorithm in ("online_attention", "flash_attention"):
                        buffering_options = (False, True) if algorithm == "flash_attention" else (False,)
                        for double_buffer in buffering_options:
                            mapping = self.Mapping(
                                q_tile,
                                kv_tile,
                                head_tile,
                                online_candidate.mapping["row_group"],
                                online_candidate.mapping["lmul"],
                                double_buffer,
                            )
                            candidate = self._fused_candidate(
                                pcb_module,
                                algorithm,
                                mapping,
                                online_candidate,
                                tile_candidates,
                            )
                            if candidate is not None:
                                candidates.append(candidate)
        candidates = deduplicate_candidates(candidates)
        if not candidates:
            raise ValueError("Attention candidate enumeration returned no candidates")
        return candidates

    def _materialized_latency(self, pcb_module, compile_mode, provider, stage, hardware, trial_sink):
        kwargs = {
            "provider": provider,
            "stage": stage,
            "hardware": hardware,
            "trial_sink": trial_sink,
        }
        qk = self.qk_matmul.compile_and_simulate(pcb_module, compile_mode, **kwargs)
        pv = self.pv_matmul.compile_and_simulate(pcb_module, compile_mode, **kwargs)
        softmax = self.softmax.compile_and_simulate(
            pcb_module, compile_mode, **kwargs
        )
        overhead = pcb_module.compute_module.overhead
        breakdown = {
            "qk": qk + overhead.matmul,
            "softmax": softmax + overhead.softmax,
            "pv": pv + overhead.matmul,
            "io": 0.0,
        }
        return sum(breakdown.values()), breakdown

    @staticmethod
    def _array_flops_per_cycle(pcb_module):
        compute = pcb_module.compute_module
        if hasattr(compute, "total_systolic_array_flops"):
            return compute.total_systolic_array_flops / compute.clock_freq
        core = compute.core
        array = core.systolic_array
        return (
            compute.core_count
            * core.systolic_array_count
            * array.mac_per_cycle
            * 2
            * array.array_height
            * array.array_width
        )

    @staticmethod
    def _vector_flops_per_cycle(pcb_module):
        compute = pcb_module.compute_module
        if hasattr(compute, "total_vector_flops_per_cycle"):
            return compute.total_vector_flops_per_cycle
        return (
            compute.core.vector_unit.total_vector_flops_per_cycle
            * compute.core_count
        )

    @staticmethod
    def _array_tile_cycles(pcb_module, m, n, k):
        """Estimate one GEMM tile with explicit array padding and wavefront cost."""
        if min(int(m), int(n), int(k)) < 1:
            return 0
        compute = pcb_module.compute_module
        core = compute.core
        array = core.systolic_array
        height = max(1, int(array.array_height))
        width = max(1, int(array.array_width))
        macs_per_cycle = max(1.0, float(array.mac_per_cycle))
        padded_m = ceil(int(m) / height) * height
        padded_n = ceil(int(n) / width) * width
        useful_cycles = ceil(
            padded_m
            * padded_n
            * int(k)
            / (height * width * macs_per_cycle)
        )
        # A systolic tile still pays the array fill/drain wavefront even when
        # the arithmetic work is small. This captures boundary utilization
        # without invoking the legacy Matmul simulator or its ScaleSim fallback.
        return max(1, useful_cycles + max(height, width) - 1)

    @staticmethod
    def _bytes_to_cycles(byte_count, bytes_per_cycle):
        if byte_count <= 0:
            return 0
        return ceil(byte_count / max(1.0, float(bytes_per_cycle)))

    def _online_tile_events(self, pcb_module, candidate, mapping):
        """Build physical L1 stripe events inside each logical L2 tile."""
        q_tiles = ceil(self.M / mapping.q_tile_M)
        kv_tiles = ceil(self.N / mapping.kv_tile_N)
        q_segments = self._tile_segments(self.M, mapping.q_tile_M, mapping.l1_q_tile_M)
        kv_segments = self._tile_segments(self.N, mapping.kv_tile_N, mapping.l1_kv_tile_N)
        word_size = self.data_type.word_size
        compute = pcb_module.compute_module
        hbm_bpc = pcb_module.io_module.bandwidth / compute.clock_freq
        l2_bpc = compute.l2_bandwidth_per_cycle
        online = candidate.strategy["algorithm"] == "online_attention"
        staged = mapping.storage_policy == "l2_staged"
        l2_capacity = max(0, int(compute.l2_size))
        resources = self._working_set(mapping, not online)
        l2_slot_bytes = int(resources.get("l2_slot_bytes", 0))
        kv_full_bytes = self.batch_heads * self.N * self.K * word_size * 2
        active_slot_count = int(
            candidate.resource_requirements.get("l2_slot_count", 0)
        )
        if active_slot_count < 1:
            active_slot_count = min(
                self.batch_heads * len(q_segments),
                max(1, int(compute.core_count)),
            )
        kv_resident = (
            kv_full_bytes + l2_slot_bytes * active_slot_count <= l2_capacity
        )
        # The Online probability stripe is already part of l2_slot_bytes.
        intermediate_in_hbm = online and l2_slot_bytes > l2_capacity
        events = []
        totals = {
            "qk": 0, "pv": 0, "softmax": 0, "output_update": 0,
            "input": 0, "middle": 0, "output": 0,
            "state_load": 0, "state_store": 0,
            "hbm_bytes": 0, "l2_bytes": 0, "hbm_cycles": 0, "l2_cycles": 0,
            "l2_staging_bytes": 0, "state_spill_bytes": 0,
        }
        variant_name = candidate.strategy["kernel_variant"]
        tile_profile_cache = {}

        def tile_profile(q_rows, kv_cols, first_kv):
            profile_key = (int(q_rows), int(kv_cols), bool(first_kv))
            cached = tile_profile_cache.get(profile_key)
            if cached is not None:
                return cached
            operator = OnlineSoftmaxUpdate(
                self.data_type, cost_model=self.SOFTMAX_COST_MODEL
            )
            operator(
                Tensor([q_rows, kv_cols], self.data_type),
                scale=self.scale,
                first=first_kv,
            )
            subcandidate = operator.resolve_variant_candidate(
                pcb_module, variant_name, allow_fallback_only=True
            )
            if subcandidate is None:
                raise ProviderProtocolError(
                    "Online Softmax variant is invalid for a physical residual stripe"
                )
            profile = (
                ceil(operator.simulate(pcb_module, subcandidate)),
                estimate_analytical_softmax_cycles(
                    q_rows * kv_cols, pcb_module
                )
                + estimate_analytical_online_state_cycles(
                    q_rows, first_kv, pcb_module
                ),
            )
            tile_profile_cache[profile_key] = profile
            return profile

        vector_flops = max(1.0, float(compute.core.vector_unit.total_vector_flops_per_cycle))
        array_count = max(1, int(compute.core_count)) * max(1, int(compute.core.systolic_array_count))
        vector_count = max(1, int(compute.core_count))
        for q_segment_index, (q_index, q_inner, _, q_rows) in enumerate(q_segments):
            for head in range(self.batch_heads):
                q_bytes = q_rows * self.K * word_size
                sequence = (head, q_index, q_inner)
                for kv_index, (kv_outer, kv_inner, _, kv_cols) in enumerate(kv_segments):
                    kv_bytes = kv_cols * self.K * word_size * 2
                    output_bytes = q_rows * self.K * 4
                    state_bytes = q_rows * 2 * 4
                    first_kv = kv_index == 0
                    last_kv = kv_index == len(kv_segments) - 1
                    (
                        softmax_cycles,
                        priority_softmax_cycles,
                    ) = tile_profile(
                        q_rows, kv_cols, first_kv
                    )
                    update_flops = q_rows * self.K * (1 if first_kv else 2)
                    update_cycles = ceil(update_flops / vector_flops)
                    qk_cycles = self._array_tile_cycles(pcb_module, q_rows, kv_cols, self.K)
                    pv_cycles = self._array_tile_cycles(pcb_module, q_rows, self.K, kv_cols)

                    hbm_input = q_bytes if first_kv else 0
                    hbm_input += kv_bytes if not kv_resident or q_segment_index == 0 else 0
                    l2_q_bytes = q_bytes if staged or first_kv else 0
                    state_load_bytes = state_bytes + output_bytes if staged and not first_kv else 0
                    state_store_bytes = state_bytes + output_bytes if staged and not last_kv else 0
                    l2_input = l2_q_bytes + kv_bytes
                    priority_l2_input = (q_bytes if first_kv else 0) + kv_bytes
                    middle_bytes = q_rows * kv_cols * 4 if online else 0
                    hbm_middle = middle_bytes if intermediate_in_hbm else 0
                    l2_middle = middle_bytes
                    hbm_output = output_bytes if last_kv else 0
                    l2_output = output_bytes if last_kv else 0
                    input_hbm_cycles = self._bytes_to_cycles(hbm_input, hbm_bpc)
                    input_l2_cycles = self._bytes_to_cycles(l2_input, l2_bpc)
                    priority_input_l2_cycles = self._bytes_to_cycles(priority_l2_input, l2_bpc)
                    softmax_state_load_cycles = self._bytes_to_cycles(
                        state_bytes if staged and not first_kv else 0, l2_bpc
                    )
                    softmax_state_store_cycles = self._bytes_to_cycles(
                        state_bytes if staged and not last_kv else 0, l2_bpc
                    )
                    output_state_load_cycles = self._bytes_to_cycles(
                        output_bytes if staged and not first_kv else 0, l2_bpc
                    )
                    output_state_store_cycles = self._bytes_to_cycles(
                        output_bytes if staged and not last_kv else 0, l2_bpc
                    )
                    state_load_cycles = (
                        softmax_state_load_cycles + output_state_load_cycles
                    )
                    state_store_cycles = (
                        softmax_state_store_cycles + output_state_store_cycles
                    )
                    middle_hbm_cycles = self._bytes_to_cycles(hbm_middle * 2, hbm_bpc)
                    middle_l2_cycles = self._bytes_to_cycles(l2_middle * 2, l2_bpc)
                    output_hbm_cycles = self._bytes_to_cycles(hbm_output, hbm_bpc)
                    output_l2_cycles = self._bytes_to_cycles(l2_output, l2_bpc)
                    event = {
                        "sequence": sequence, "l2_staged": staged,
                        "l2_slot": (
                            (q_segment_index * self.batch_heads + head)
                            % max(1, active_slot_count)
                        ),
                        "l2_slot_count": int(candidate.resource_requirements.get("l2_slot_count", 0)),
                        "head": head, "q_index": q_index, "q_inner_index": q_inner,
                        "kv_index": kv_index, "kv_outer_index": kv_outer,
                        "kv_inner_index": kv_inner, "q_rows": q_rows, "kv_cols": kv_cols,
                        "array_count": array_count, "vector_count": vector_count,
                        "qk": qk_cycles, "pv": pv_cycles, "softmax": softmax_cycles,
                        "priority_softmax": priority_softmax_cycles,
                        "output_update": update_cycles,
                        "compute": qk_cycles + softmax_cycles + pv_cycles + update_cycles,
                        "input": input_hbm_cycles + input_l2_cycles,
                        "middle": middle_hbm_cycles + middle_l2_cycles,
                        "output": output_hbm_cycles + output_l2_cycles,
                        "state_load": state_load_cycles, "state_store": state_store_cycles,
                        "input_hbm_cycles": input_hbm_cycles,
                        "input_l2_cycles": input_l2_cycles,
                        "priority_input_l2_cycles": priority_input_l2_cycles,
                        "middle_hbm_cycles": middle_hbm_cycles,
                        "middle_l2_cycles": middle_l2_cycles,
                        "output_hbm_cycles": output_hbm_cycles,
                        "output_l2_cycles": output_l2_cycles,
                        "softmax_state_load_l2_cycles": softmax_state_load_cycles,
                        "softmax_state_store_l2_cycles": softmax_state_store_cycles,
                        "output_state_load_l2_cycles": output_state_load_cycles,
                        "output_state_store_l2_cycles": output_state_store_cycles,
                        "hbm_bytes": hbm_input + hbm_middle * 2 + hbm_output,
                        "l2_bytes": l2_input + l2_middle * 2 + l2_output + state_load_bytes + state_store_bytes,
                        "l2_staging_bytes": (
                            (q_bytes if staged and not first_kv else 0)
                            + state_load_bytes + state_store_bytes
                        ),
                        "state_spill_bytes": state_load_bytes + state_store_bytes,
                        "hbm_cycles": input_hbm_cycles + middle_hbm_cycles + output_hbm_cycles,
                        "l2_cycles": (
                            input_l2_cycles + middle_l2_cycles + output_l2_cycles
                            + state_load_cycles + state_store_cycles
                        ),
                    }
                    events.append(event)
                    for name in totals:
                        if name in event:
                            totals[name] += event[name]
        totals["q_stripes"] = len(q_segments)
        totals["kv_stripes"] = len(kv_segments)
        return events, totals, q_tiles, kv_tiles

    @staticmethod
    def _schedule_tile_events(events, double_buffer, return_trace=False):
        """Schedule the tile DAG with a duration-independent list order."""
        if not events:
            return (0, {"dependencies": {}, "finish": {}}) if return_trace else 0
        array_count = max(1, int(events[0].get("array_count", 1)))
        vector_count = max(1, int(events[0].get("vector_count", 1)))
        buffer_count = 2 if double_buffer else 1
        stage_order = {
            "input": 0,
            "qk": 1,
            "softmax_state_load": 2,
            "softmax": 3,
            "softmax_state_store": 4,
            "middle": 5,
            "pv": 6,
            "output_state_load": 7,
            "output_update": 8,
            "output_state_store": 9,
            "output": 10,
        }
        task_deps = {}
        dependents = {}
        task_info = {}

        def add_task(task, stage, event, dependencies):
            task_deps[task] = set(dependencies)
            task_info[task] = (stage, event)
            for dependency in dependencies:
                dependents.setdefault(dependency, []).append(task)

        sequences = {}
        for index, event in enumerate(events):
            sequences.setdefault(event.get("sequence", index), []).append(index)
        sequence_order = list(sequences)
        sequence_position = {
            sequence: position
            for position, sequence in enumerate(sequence_order)
        }
        for index, event in enumerate(events):
            sequence = event.get("sequence", index)
            sequence_events = sequences[sequence]
            position = sequence_events.index(index)
            previous = sequence_events[position - 1] if position else None
            last_in_sequence = position == len(sequence_events) - 1
            buffer_previous = (
                sequence_events[position - buffer_count]
                if position >= buffer_count
                else None
            )
            input_deps = [] if buffer_previous is None else [("output", buffer_previous)]
            # Staged sequences occupy finite L2 slots. Reusing a slot waits for
            # the previous owner's final output/state store instead of assuming
            # that every head and Q stripe can be resident concurrently.
            slot_count = int(event.get("l2_slot_count", 0))
            seq_position = sequence_position[sequence]
            if (
                event.get("l2_staged", False)
                and position == 0
                and slot_count > 0
                and seq_position >= slot_count
            ):
                prior_sequence = sequence_order[seq_position - slot_count]
                input_deps.append(("output", sequences[prior_sequence][-1]))
            add_task(("input", index), "input", event, input_deps)
            add_task(("qk", index), "qk", event, [("input", index)])
            softmax_deps = [("qk", index)]
            if event.get("l2_staged", False) and previous is not None:
                add_task(
                    ("softmax_state_load", index),
                    "softmax_state_load",
                    event,
                    [("softmax_state_store", previous)],
                )
                softmax_deps.append(("softmax_state_load", index))
            elif previous is not None:
                softmax_deps.append(("softmax", previous))
            add_task(("softmax", index), "softmax", event, softmax_deps)
            softmax_store = None
            if event.get("l2_staged", False) and not last_in_sequence:
                softmax_store = ("softmax_state_store", index)
                add_task(
                    softmax_store,
                    "softmax_state_store",
                    event,
                    [("softmax", index)],
                )
            add_task(("middle", index), "middle", event, [("softmax", index)])
            add_task(("pv", index), "pv", event, [("middle", index)])
            update_deps = [("pv", index)]
            if event.get("l2_staged", False) and previous is not None:
                add_task(
                    ("output_state_load", index),
                    "output_state_load",
                    event,
                    [("output_state_store", previous)],
                )
                update_deps.append(("output_state_load", index))
            elif previous is not None:
                update_deps.append(("output_update", previous))
            add_task(
                ("output_update", index),
                "output_update",
                event,
                update_deps,
            )
            output_deps = [("output_update", index)]
            if softmax_store is not None:
                output_deps.append(softmax_store)
            if event.get("l2_staged", False) and not last_in_sequence:
                output_store = ("output_state_store", index)
                add_task(
                    output_store,
                    "output_state_store",
                    event,
                    [("output_update", index)],
                )
                output_deps.append(output_store)
            add_task(
                ("output", index),
                "output",
                event,
                output_deps,
            )

        def duration_for(stage, event, priority):
            if stage == "input":
                return max(
                    event.get("input_hbm_cycles", event.get("input", 0)),
                    (
                        event.get("priority_input_l2_cycles", 0)
                        if priority
                        else event.get("input_l2_cycles", 0)
                    ),
                )
            if stage == "qk":
                return event.get("qk", 0)
            if stage == "softmax_state_load":
                return event.get("softmax_state_load_l2_cycles", 0)
            if stage == "softmax":
                if priority:
                    actual = event.get("softmax", 0)
                    return (
                        event.get("priority_softmax", 1)
                        if actual > 0
                        else 0
                    )
                return event.get("softmax", 0)
            if stage == "middle":
                return max(
                    event.get("middle_hbm_cycles", 0),
                    event.get("middle_l2_cycles", 0),
                )
            if stage == "softmax_state_store":
                return event.get("softmax_state_store_l2_cycles", 0)
            if stage == "pv":
                return event.get("pv", 0)
            if stage == "output_state_load":
                return event.get("output_state_load_l2_cycles", 0)
            if stage == "output_update":
                return event.get("output_update", 0)
            if stage == "output_state_store":
                return event.get("output_state_store_l2_cycles", 0)
            return max(
                event.get("output_hbm_cycles", 0),
                event.get("output_l2_cycles", 0),
            )

        def reserve_engine(engine_heap, ready, duration):
            if duration <= 0:
                return int(ready)
            free_at = heapq.heappop(engine_heap)
            finish = max(int(ready), int(free_at)) + int(duration)
            heapq.heappush(engine_heap, finish)
            return finish

        def reserve_task(task, ready, state, priority):
            stage, event = task_info[task]
            duration = max(0, int(duration_for(stage, event, priority)))
            if stage in ("qk", "pv"):
                return reserve_engine(state["array"], ready, duration)
            if stage in ("softmax", "output_update"):
                return reserve_engine(state["vector"], ready, duration)
            state_memory_task = stage in (
                "softmax_state_load",
                "softmax_state_store",
                "output_state_load",
                "output_state_store",
            )
            if state_memory_task and duration <= 0:
                return int(ready)
            finish = max(int(ready), state["memory"]) + duration
            state["memory"] = finish
            return finish

        def new_state():
            arrays = [0] * array_count
            vectors = [0] * vector_count
            heapq.heapify(arrays)
            heapq.heapify(vectors)
            return {"array": arrays, "vector": vectors, "memory": 0}

        # Build the staged DAG list order with variant-independent Softmax and
        # input durations. Reusing it for actual durations prevents candidate
        # variants from changing arbitration solely because they are faster.
        remaining = {task: len(deps) for task, deps in task_deps.items()}
        priority_ready = {task: 0 for task in task_deps}
        priority_heap = []
        priority_state = new_state()
        task_order = []
        for task, count in remaining.items():
            if count == 0:
                stage, _ = task_info[task]
                heapq.heappush(
                    priority_heap,
                    (0, stage_order[stage], task[1], task),
                )

        while priority_heap:
            _, _, _, task = heapq.heappop(priority_heap)
            finish = reserve_task(
                task, priority_ready[task], priority_state, True
            )
            task_order.append(task)
            for dependent in dependents.get(task, ()):
                remaining[dependent] -= 1
                priority_ready[dependent] = max(
                    priority_ready[dependent], finish
                )
                if remaining[dependent] == 0:
                    stage, _ = task_info[dependent]
                    heapq.heappush(
                        priority_heap,
                        (
                            priority_ready[dependent],
                            stage_order[stage],
                            dependent[1],
                            dependent,
                        ),
                    )

        if len(task_order) != len(task_info):
            raise ProviderProtocolError("Attention tile dependency graph is cyclic")

        actual_state = new_state()
        finish_times = {}
        for task in task_order:
            ready = max(
                (finish_times[dependency] for dependency in task_deps[task]),
                default=0,
            )
            finish_times[task] = reserve_task(
                task, ready, actual_state, False
            )
        makespan = int(max(finish_times.values(), default=0))
        if return_trace:
            return makespan, {
                "dependencies": {
                    task: set(dependencies)
                    for task, dependencies in task_deps.items()
                },
                "finish": dict(finish_times),
                "order": list(task_order),
            }
        return makespan

    def _online_cycles(self, pcb_module, candidate):
        mapping = candidate.mapping_object
        if mapping is None:
            mapping = self.Mapping(**candidate.mapping)
            candidate.mapping_object = mapping
        events, totals, q_tiles, kv_tiles = self._online_tile_events(
            pcb_module, candidate, mapping
        )
        flash = candidate.strategy["algorithm"] == "flash_attention"
        double_buffer = flash and mapping.is_kv_double_buffering
        scheduled_cycles = self._schedule_tile_events_fast(
            events, double_buffer
        )
        launch_count = 1 if flash else 2
        launch_cycles = ceil(
            pcb_module.compute_module.overhead.matmul
            * pcb_module.compute_module.clock_freq
        ) * launch_count
        visible_io = max(
            0,
            scheduled_cycles
            - totals["qk"]
            - totals["pv"]
            - totals["softmax"]
            - totals["output_update"],
        )
        compute_busy = (
            totals["qk"]
            + totals["pv"]
            + totals["softmax"]
            + totals["output_update"]
        )
        overlap = max(0, compute_busy - scheduled_cycles)
        total_cycles = scheduled_cycles + launch_cycles
        return {
            "total": int(max(1, total_cycles)),
            "qk": int(totals["qk"]),
            "pv": int(totals["pv"]),
            "softmax": int(totals["softmax"]),
            "output_update": int(totals["output_update"]),
            "io": int(visible_io),
            "overlap": int(overlap),
            "launch": int(launch_cycles),
            "launch_count": launch_count,
            "q_tiles": q_tiles,
            "kv_tiles": kv_tiles,
            # Keep global_bytes as the legacy on-chip traffic comparison key.
            "global_bytes": int(totals["l2_bytes"]),
            "hbm_bytes": int(totals["hbm_bytes"]),
            "l2_bytes": int(totals["l2_bytes"]),
            "hbm_cycles": int(totals["hbm_cycles"]),
            "l2_cycles": int(totals["l2_cycles"]),
            "l2_staging_bytes": int(totals["l2_staging_bytes"]),
            "state_spill_bytes": int(totals["state_spill_bytes"]),
            "state_load_cycles": int(totals["state_load"]),
            "state_store_cycles": int(totals["state_store"]),
            "scheduled_cycles": int(scheduled_cycles),
            "double_buffered": bool(double_buffer),
            "storage_policy": mapping.storage_policy,
            "l1_q_tile_M": mapping.l1_q_tile_M,
            "l1_kv_tile_N": mapping.l1_kv_tile_N,
            "q_stripes": int(totals["q_stripes"]),
            "kv_stripes": int(totals["kv_stripes"]),
            "l2_slot_count": int(
                candidate.resource_requirements.get("l2_slot_count", 0)
            ),
            "l2_slot_bytes": int(
                candidate.resource_requirements.get("l2_slot_bytes", 0)
            ),
        }

    @staticmethod
    def _schedule_tile_events_fast(events, double_buffer, return_trace=False):
        """Run the reference DAG policy with compact integer task storage."""
        if not events:
            return 0
        stage_count = 11
        event_count = len(events)
        task_count = event_count * stage_count
        array_count = max(1, int(events[0].get("array_count", 1)))
        vector_count = max(1, int(events[0].get("vector_count", 1)))
        buffer_count = 2 if double_buffer else 1

        previous = array("i", [-1]) * event_count
        following = array("i", [-1]) * event_count
        buffer_previous = array("i", [-1]) * event_count
        buffer_following = array("i", [-1]) * event_count
        slot_previous = array("i", [-1]) * event_count
        slot_following = array("i", [-1]) * event_count
        ranges = []
        range_start = 0
        while range_start < event_count:
            sequence = events[range_start].get("sequence", range_start)
            range_end = range_start + 1
            while (
                range_end < event_count
                and events[range_end].get("sequence", range_end) == sequence
            ):
                range_end += 1
            ranges.append((range_start, range_end))
            for event_index in range(range_start, range_end):
                if event_index > range_start:
                    previous[event_index] = event_index - 1
                if event_index + 1 < range_end:
                    following[event_index] = event_index + 1
                if event_index - buffer_count >= range_start:
                    buffer_previous[event_index] = event_index - buffer_count
                if event_index + buffer_count < range_end:
                    buffer_following[event_index] = event_index + buffer_count
            range_start = range_end

        slot_count = int(events[0].get("l2_slot_count", 0))
        if events[0].get("l2_staged", False) and slot_count > 0:
            for sequence_index in range(slot_count, len(ranges)):
                current_start = ranges[sequence_index][0]
                prior_end = ranges[sequence_index - slot_count][1] - 1
                slot_previous[current_start] = prior_end
                slot_following[prior_end] = current_start

        # Stage codes match the reference stage_order:
        # input, qk, softmax-load, softmax, softmax-store, middle, pv,
        # output-load, output-update, output-store, output.
        missing = 255
        remaining = bytearray([missing]) * task_count
        priority_ready = array("q", [0]) * task_count
        actual_finish = array("q", [0]) * task_count

        def task_id(event_index, stage):
            return event_index * stage_count + stage

        present_count = 0

        def add_task(event_index, stage, dependency_count):
            nonlocal present_count
            remaining[task_id(event_index, stage)] = dependency_count
            present_count += 1

        for event_index, event in enumerate(events):
            staged = event.get("l2_staged", False)
            prior = previous[event_index]
            later = following[event_index]
            input_dependencies = {
                dependency
                for dependency in (
                    buffer_previous[event_index], slot_previous[event_index]
                )
                if dependency >= 0
            }
            add_task(event_index, 0, len(input_dependencies))
            add_task(event_index, 1, 1)
            if staged and prior >= 0:
                add_task(event_index, 2, 1)
            add_task(event_index, 3, 2 if prior >= 0 else 1)
            if staged and later >= 0:
                add_task(event_index, 4, 1)
            add_task(event_index, 5, 1)
            add_task(event_index, 6, 1)
            if staged and prior >= 0:
                add_task(event_index, 7, 1)
            add_task(event_index, 8, 2 if prior >= 0 else 1)
            if staged and later >= 0:
                add_task(event_index, 9, 1)
            output_dependencies = 1
            if staged and later >= 0:
                output_dependencies += 2
            add_task(event_index, 10, output_dependencies)

        def notify(target_event, target_stage, finish, ready_heap):
            if target_event < 0:
                return
            target = task_id(target_event, target_stage)
            if remaining[target] == missing:
                return
            remaining[target] -= 1
            if finish > priority_ready[target]:
                priority_ready[target] = finish
            if remaining[target] == 0:
                heapq.heappush(
                    ready_heap,
                    (priority_ready[target], target_stage, target_event),
                )

        def notify_successors(event_index, stage, finish, ready_heap):
            staged = events[event_index].get("l2_staged", False)
            later = following[event_index]
            if stage == 0:
                notify(event_index, 1, finish, ready_heap)
            elif stage == 1:
                notify(event_index, 3, finish, ready_heap)
            elif stage == 2:
                notify(event_index, 3, finish, ready_heap)
            elif stage == 3:
                notify(event_index, 5, finish, ready_heap)
                if staged and later >= 0:
                    notify(event_index, 4, finish, ready_heap)
                elif later >= 0:
                    notify(later, 3, finish, ready_heap)
            elif stage == 4:
                notify(later, 2, finish, ready_heap)
                notify(event_index, 10, finish, ready_heap)
            elif stage == 5:
                notify(event_index, 6, finish, ready_heap)
            elif stage == 6:
                notify(event_index, 8, finish, ready_heap)
            elif stage == 7:
                notify(event_index, 8, finish, ready_heap)
            elif stage == 8:
                if staged and later >= 0:
                    notify(event_index, 9, finish, ready_heap)
                elif later >= 0:
                    notify(later, 8, finish, ready_heap)
                notify(event_index, 10, finish, ready_heap)
            elif stage == 9:
                notify(later, 7, finish, ready_heap)
                notify(event_index, 10, finish, ready_heap)
            else:
                targets = {
                    target
                    for target in (
                        buffer_following[event_index], slot_following[event_index]
                    )
                    if target >= 0
                }
                for target in targets:
                    notify(target, 0, finish, ready_heap)

        def duration(event, stage, priority):
            if stage == 0:
                return max(
                    event.get("input_hbm_cycles", event.get("input", 0)),
                    event.get(
                        "priority_input_l2_cycles" if priority else "input_l2_cycles",
                        0,
                    ),
                )
            if stage == 1:
                return event.get("qk", 0)
            if stage == 2:
                return event.get("softmax_state_load_l2_cycles", 0)
            if stage == 3:
                actual = event.get("softmax", 0)
                return event.get("priority_softmax", 1) if priority and actual > 0 else actual
            if stage == 4:
                return event.get("softmax_state_store_l2_cycles", 0)
            if stage == 5:
                return max(
                    event.get("middle_hbm_cycles", 0),
                    event.get("middle_l2_cycles", 0),
                )
            if stage == 6:
                return event.get("pv", 0)
            if stage == 7:
                return event.get("output_state_load_l2_cycles", 0)
            if stage == 8:
                return event.get("output_update", 0)
            if stage == 9:
                return event.get("output_state_store_l2_cycles", 0)
            return max(
                event.get("output_hbm_cycles", 0),
                event.get("output_l2_cycles", 0),
            )

        def reserve_engine(engine_heap, ready, duration):
            if duration <= 0:
                return int(ready)
            free_at = heapq.heappop(engine_heap)
            finish = max(int(ready), int(free_at)) + int(duration)
            heapq.heappush(engine_heap, finish)
            return finish

        def new_resource_state():
            arrays = [0] * array_count
            vectors = [0] * vector_count
            heapq.heapify(arrays)
            heapq.heapify(vectors)
            return [arrays, vectors, 0]

        def reserve(event, stage, ready, state, priority):
            task_duration = max(0, int(duration(event, stage, priority)))
            if stage in (1, 6):
                return reserve_engine(state[0], ready, task_duration)
            if stage in (3, 8):
                return reserve_engine(state[1], ready, task_duration)
            if stage in (2, 4, 7, 9) and task_duration <= 0:
                return int(ready)
            finish = max(int(ready), state[2]) + task_duration
            state[2] = finish
            return finish

        ready_heap = []
        for event_index in range(event_count):
            for stage in range(stage_count):
                task = task_id(event_index, stage)
                if remaining[task] == 0:
                    heapq.heappush(ready_heap, (0, stage, event_index))

        priority_state = new_resource_state()
        task_order = array("I")
        processed = 0
        while ready_heap:
            ready, stage, event_index = heapq.heappop(ready_heap)
            finish = reserve(
                events[event_index], stage, ready, priority_state, True
            )
            task_order.append(task_id(event_index, stage))
            processed += 1
            notify_successors(event_index, stage, finish, ready_heap)
        if processed != present_count:
            raise ProviderProtocolError("Attention tile dependency graph is cyclic")

        def predecessor_finish(event_index, stage):
            prior = previous[event_index]
            values = []
            if stage == 0:
                values.extend(
                    actual_finish[task_id(dependency, 10)]
                    for dependency in {
                        value
                        for value in (
                            buffer_previous[event_index],
                            slot_previous[event_index],
                        )
                        if value >= 0
                    }
                )
            elif stage == 1:
                values.append(actual_finish[task_id(event_index, 0)])
            elif stage == 2:
                values.append(actual_finish[task_id(prior, 4)])
            elif stage == 3:
                values.append(actual_finish[task_id(event_index, 1)])
                if prior >= 0:
                    values.append(
                        actual_finish[task_id(event_index, 2)]
                        if events[event_index].get("l2_staged", False)
                        else actual_finish[task_id(prior, 3)]
                    )
            elif stage == 4:
                values.append(actual_finish[task_id(event_index, 3)])
            elif stage == 5:
                values.append(actual_finish[task_id(event_index, 3)])
            elif stage == 6:
                values.append(actual_finish[task_id(event_index, 5)])
            elif stage == 7:
                values.append(actual_finish[task_id(prior, 9)])
            elif stage == 8:
                values.append(actual_finish[task_id(event_index, 6)])
                if prior >= 0:
                    values.append(
                        actual_finish[task_id(event_index, 7)]
                        if events[event_index].get("l2_staged", False)
                        else actual_finish[task_id(prior, 8)]
                    )
            elif stage == 9:
                values.append(actual_finish[task_id(event_index, 8)])
            else:
                values.append(actual_finish[task_id(event_index, 8)])
                if events[event_index].get("l2_staged", False) and following[event_index] >= 0:
                    values.append(actual_finish[task_id(event_index, 4)])
                    values.append(actual_finish[task_id(event_index, 9)])
            return max(values, default=0)

        actual_state = new_resource_state()
        makespan = 0
        for task in task_order:
            event_index, stage = divmod(task, stage_count)
            finish = reserve(
                events[event_index],
                stage,
                predecessor_finish(event_index, stage),
                actual_state,
                False,
            )
            actual_finish[task] = finish
            if finish > makespan:
                makespan = finish
        if return_trace:
            return int(makespan), {
                "order": list(task_order),
                "finish": actual_finish,
            }
        return int(makespan)


    def _advanced_latency(self, pcb_module, candidate):
        cycles = self._online_cycles(pcb_module, candidate)
        frequency = pcb_module.compute_module.clock_freq
        # The scheduler already accounts for any hidden input prefetch.
        # Keep component busy times intact and expose only the unscheduled I/O
        # in the breakdown instead of scaling all components by an ideal ratio.
        breakdown_cycles = {
            "qk": cycles["qk"],
            "pv": cycles["pv"],
            "softmax": cycles["softmax"],
            "output_update": cycles["output_update"],
            "io": cycles["io"],
            # Busy times are additive, while the scheduler returns a
            # makespan. Expose overlap explicitly so the breakdown still
            # reconciles to the reported latency.
            "overlap": -cycles.get("overlap", 0),
        }
        launch_per_kernel = cycles["launch"] / cycles["launch_count"]
        breakdown_cycles["qk"] += launch_per_kernel
        if cycles["launch_count"] == 2:
            breakdown_cycles["pv"] += launch_per_kernel
        breakdown = {
            name: value / frequency for name, value in breakdown_cycles.items()
        }
        return cycles["total"] / frequency, cycles, breakdown

    def evaluate_transfer_candidate(self, pcb_module, candidate, trial_sink=None):
        candidate = self.validate_candidate(candidate)
        if not isinstance(candidate.strategy, dict):
            raise ProviderProtocolError("Attention candidate requires structured strategy")
        algorithm = candidate.strategy.get("algorithm")
        if algorithm not in ("materialized_attention", "online_attention", "flash_attention"):
            raise ProviderProtocolError("unsupported Attention strategy: {}".format(algorithm))
        if algorithm == "materialized_attention":
            latency, breakdown = self._materialized_latency(
                pcb_module,
                getattr(self, "_compile_mode", "heuristic-GPU"),
                getattr(self, "_provider", None),
                getattr(self, "_stage", None),
                getattr(self, "_hardware", None),
                trial_sink,
            )
            cycle_count = ceil(latency * pcb_module.compute_module.clock_freq)
            cycles = {"total": cycle_count, "model": "legacy_three_operator_sum"}
        else:
            latency, cycles, breakdown = self._advanced_latency(
                pcb_module, candidate
            )
            cycle_count = cycles["total"]
        recorder = get_active_recorder()
        if recorder is not None and self.recording_name is not None:
            recorder.record_operator_mapping_trial(
                operator_name=self.recording_name,
                operator_type="FusedAttention",
                execution_kind=candidate.execution_kind,
                graph=candidate.graph,
                mapping=candidate.mapping,
                cycle_count=cycle_count,
                raw_local_latency_s=latency,
                candidate=candidate,
            )
        if trial_sink is not None:
            trial_sink(candidate.to_dict(), latency, cycle_count, candidate.strategy)
        self.last_cycle_breakdown = cycles
        self.last_latency_breakdown = breakdown
        self.best_mapping = candidate.mapping
        self.best_cycle_count = cycle_count
        self.best_latency = latency
        self.execution_kind = candidate.execution_kind
        self.selected_strategy = candidate.strategy
        self.latency = latency
        return latency

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode="heuristic-GPU",
        provider=None,
        stage=None,
        hardware=None,
        trial_sink=None,
        operator_name=None,
    ):
        self._compile_mode = compile_mode
        self._provider = provider
        self._stage = stage
        self._hardware = hardware
        generation_mode = (
            getattr(provider, "generation_mode", "heuristic-GPU")
            if compile_mode == "transfer-learn"
            else compile_mode
        )
        if generation_mode is None:
            generation_mode = "heuristic-GPU"
        candidates = self.enumerate_transfer_candidates(pcb_module, generation_mode)
        if compile_mode == "transfer-learn":
            if provider is None:
                raise ProviderProtocolError("transfer-learn requires a provider")
            records = [candidate.to_dict() for candidate in candidates]
            top_k = getattr(provider, "top_k", 1)
            ranked = provider.rank(
                operator_name or self.recording_name or "Attention",
                stage,
                hardware or {},
                {},
                records,
                top_k,
            )
            selected_ids = validate_ranked_ids(records, ranked, top_k)
            by_id = {candidate.candidate_id: candidate for candidate in candidates}
            candidates = [by_id[candidate_id] for candidate_id in selected_ids]

        best = None
        for candidate in candidates:
            latency = self.evaluate_transfer_candidate(
                pcb_module, candidate, trial_sink
            )
            result = (
                latency,
                candidate,
                self.best_cycle_count,
                dict(self.last_cycle_breakdown),
                dict(self.last_latency_breakdown),
            )
            if best is None or (latency, candidate.candidate_id) < (
                best[0],
                best[1].candidate_id,
            ):
                best = result
        if best is None:
            raise ProviderProtocolError("no Attention candidate was evaluated")
        (
            self.best_latency,
            selected,
            self.best_cycle_count,
            self.last_cycle_breakdown,
            self.last_latency_breakdown,
        ) = best
        self.best_mapping = selected.mapping
        self.execution_kind = selected.execution_kind
        self.selected_strategy = selected.strategy
        self.latency = self.best_latency
        return self.latency

    def roofline_model(self, pcb_module: Device):
        overhead = pcb_module.compute_module.overhead
        breakdown = {
            "qk": self.qk_matmul.roofline_model(pcb_module) + overhead.matmul,
            "softmax": self.softmax.roofline_model(pcb_module) + overhead.softmax,
            "pv": self.pv_matmul.roofline_model(pcb_module) + overhead.matmul,
            "io": 0.0,
        }
        materialized = sum(breakdown.values())
        best = (materialized, breakdown)
        for candidate in self.enumerate_transfer_candidates(pcb_module):
            if candidate.strategy["algorithm"] == "materialized_attention":
                continue
            latency, _, candidate_breakdown = self._advanced_latency(
                pcb_module, candidate
            )
            if latency < best[0]:
                best = latency, candidate_breakdown
        self.roofline_latency, self.last_latency_breakdown = best
        return self.roofline_latency
