"""Searchable materialized, online, and FlashAttention implementations."""

import heapq
import math
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
        ):
            self.q_tile_M = int(q_tile_M)
            self.kv_tile_N = int(kv_tile_N)
            self.head_dim_tile_K = int(head_dim_tile_K)
            self.row_group = int(row_group)
            self.lmul = int(lmul)
            self.is_kv_double_buffering = bool(is_kv_double_buffering)

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
        operator = OnlineSoftmaxUpdate(
            self.data_type, cost_model=self.SOFTMAX_COST_MODEL
        )
        operator(
            Tensor([q_tile, kv_tile], self.data_type),
            scale=self.scale,
            first=False,
        )
        return operator.enumerate_transfer_candidates(
            pcb_module, "exhaustive", allow_empty=True
        )

    def _online_tile_candidates(
        self, pcb_module, q_tile, kv_tile, variant_name
    ):
        """Return the selected variant for every actual residual tile shape."""
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
                    return None
                selected.append(candidate)
        return selected

    def _working_set(self, mapping, flash):
        word_size = self.data_type.word_size
        q_bytes = mapping.q_tile_M * mapping.head_dim_tile_K * word_size
        k_bytes = mapping.kv_tile_N * mapping.head_dim_tile_K * word_size
        v_bytes = mapping.kv_tile_N * self.K * word_size
        if mapping.is_kv_double_buffering:
            k_bytes *= 2
            v_bytes *= 2
        score_bytes = mapping.q_tile_M * mapping.kv_tile_N * 4
        output_bytes = mapping.q_tile_M * self.K * 4
        state_bytes = mapping.q_tile_M * 2 * 4
        # FlashAttention keeps one score tile live; online_attention also keeps
        # a probability tile because QK/softmax and PV are separate kernels.
        probability_bytes = 0 if flash else score_bytes
        return {
            "q_tile_bytes": q_bytes,
            "kv_tile_bytes": k_bytes + v_bytes,
            "score_tile_bytes": score_bytes,
            "probability_tile_bytes": probability_bytes,
            "output_accumulator_bytes": output_bytes,
            "online_state_bytes": state_bytes,
            "sram_bytes": q_bytes
            + k_bytes
            + v_bytes
            + score_bytes
            + probability_bytes
            + output_bytes
            + state_bytes,
        }

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
        resources["sram_bytes"] = max(
            resources["sram_bytes"], resources["online_softmax_l1_bytes"]
        )
        if resources["sram_bytes"] > pcb_module.compute_module.core.SRAM_size:
            return None
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
        head_tile = min(self.K, 64)
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
        """Build QK/softmax/PV events and hierarchy-specific traffic per tile."""
        q_tiles = ceil(self.M / mapping.q_tile_M)
        kv_tiles = ceil(self.N / mapping.kv_tile_N)
        word_size = self.data_type.word_size
        compute = pcb_module.compute_module
        hbm_bpc = pcb_module.io_module.bandwidth / compute.clock_freq
        l2_bpc = compute.l2_bandwidth_per_cycle
        online = candidate.strategy["algorithm"] == "online_attention"
        # KV, Q, output/state, and the active tile share L2 capacity.  A full
        # KV residency claim is valid only after reserving those other live
        # objects; otherwise the model silently assumes an impossible cache.
        l2_capacity = max(0, int(compute.l2_size))
        active_q_bytes = self.batch_heads * mapping.q_tile_M * self.K * word_size
        active_output_bytes = self.batch_heads * mapping.q_tile_M * self.K * 4
        active_state_bytes = self.batch_heads * mapping.q_tile_M * 2 * 4
        active_kv_bytes = (
            self.batch_heads * mapping.kv_tile_N * self.K * word_size * 2
        )
        if mapping.is_kv_double_buffering:
            active_kv_bytes *= 2
        active_middle_bytes = (
            self.batch_heads * mapping.q_tile_M * mapping.kv_tile_N * 4
            if online
            else 0
        )
        l2_non_kv = (
            active_q_bytes
            + active_output_bytes
            + active_state_bytes
            + active_kv_bytes
        )
        kv_full_bytes = self.batch_heads * self.N * self.K * word_size * 2
        kv_resident = kv_full_bytes + l2_non_kv <= l2_capacity
        intermediate_in_hbm = online and (
            l2_non_kv + active_middle_bytes > l2_capacity
        )
        events = []
        totals = {
            "qk": 0,
            "pv": 0,
            "softmax": 0,
            "output_update": 0,
            "input": 0,
            "middle": 0,
            "output": 0,
            "hbm_bytes": 0,
            "l2_bytes": 0,
            "hbm_cycles": 0,
            "l2_cycles": 0,
        }
        variant_name = candidate.strategy["kernel_variant"]
        vector_flops = max(
            1.0,
            float(compute.core.vector_unit.total_vector_flops_per_cycle),
        )
        array_count = max(1, int(compute.core_count)) * max(
            1, int(compute.core.systolic_array_count)
        )
        vector_count = max(1, int(compute.core_count))
        for q_index in range(q_tiles):
            q_rows = min(mapping.q_tile_M, self.M - q_index * mapping.q_tile_M)
            for head in range(self.batch_heads):
                q_bytes = q_rows * self.K * word_size
                for kv_index in range(kv_tiles):
                    kv_cols = min(
                        mapping.kv_tile_N, self.N - kv_index * mapping.kv_tile_N
                    )
                    kv_bytes = kv_cols * self.K * word_size * 2
                    output_bytes = q_rows * self.K * 4
                    operator = OnlineSoftmaxUpdate(
                        self.data_type, cost_model=self.SOFTMAX_COST_MODEL
                    )
                    operator(
                        Tensor([q_rows, kv_cols], self.data_type),
                        scale=self.scale,
                        first=kv_index == 0,
                    )
                    subcandidate = operator.resolve_variant_candidate(
                        pcb_module,
                        variant_name,
                        allow_fallback_only=True,
                    )
                    if subcandidate is None:
                        raise ProviderProtocolError(
                            "Online Softmax variant is invalid for a residual tile"
                        )
                    softmax_cycles = ceil(operator.simulate(pcb_module, subcandidate))
                    priority_softmax_cycles = (
                        estimate_analytical_softmax_cycles(
                            q_rows * kv_cols, pcb_module
                        )
                        + estimate_analytical_online_state_cycles(
                            q_rows, kv_index == 0, pcb_module
                        )
                    )
                    update_flops = q_rows * self.K * (1 if kv_index == 0 else 2)
                    update_cycles = ceil(update_flops / vector_flops)
                    qk_cycles = self._array_tile_cycles(
                        pcb_module, q_rows, kv_cols, self.K
                    )
                    pv_cycles = self._array_tile_cycles(
                        pcb_module, q_rows, self.K, kv_cols
                    )

                    hbm_input = q_bytes if kv_index == 0 else 0
                    if kv_resident:
                        hbm_input += kv_bytes if q_index == 0 else 0
                    else:
                        hbm_input += kv_bytes
                    l2_input = (q_bytes if kv_index == 0 else 0) + kv_bytes
                    middle_bytes = q_rows * kv_cols * 4 if online else 0
                    hbm_middle = middle_bytes if intermediate_in_hbm else 0
                    l2_middle = middle_bytes
                    hbm_output = output_bytes if kv_index == kv_tiles - 1 else 0
                    l2_output = output_bytes if kv_index == kv_tiles - 1 else 0
                    input_hbm_cycles = self._bytes_to_cycles(hbm_input, hbm_bpc)
                    input_l2_cycles = self._bytes_to_cycles(l2_input, l2_bpc)
                    middle_hbm_cycles = self._bytes_to_cycles(hbm_middle * 2, hbm_bpc)
                    middle_l2_cycles = self._bytes_to_cycles(l2_middle * 2, l2_bpc)
                    output_hbm_cycles = self._bytes_to_cycles(hbm_output, hbm_bpc)
                    output_l2_cycles = self._bytes_to_cycles(l2_output, l2_bpc)
                    input_cycles = input_hbm_cycles + input_l2_cycles
                    middle_cycles = middle_hbm_cycles + middle_l2_cycles
                    output_cycles = output_hbm_cycles + output_l2_cycles
                    compute_cycles = qk_cycles + softmax_cycles + pv_cycles + update_cycles
                    event = {
                        "sequence": (head, q_index),
                        "head": head,
                        "q_index": q_index,
                        "kv_index": kv_index,
                        "array_count": array_count,
                        "vector_count": vector_count,
                        "qk": qk_cycles,
                        "pv": pv_cycles,
                        "softmax": softmax_cycles,
                        "priority_softmax": priority_softmax_cycles,
                        "output_update": update_cycles,
                        "compute": compute_cycles,
                        "input": input_cycles,
                        "middle": middle_cycles,
                        "output": output_cycles,
                        "input_hbm_cycles": input_hbm_cycles,
                        "input_l2_cycles": input_l2_cycles,
                        "middle_hbm_cycles": middle_hbm_cycles,
                        "middle_l2_cycles": middle_l2_cycles,
                        "output_hbm_cycles": output_hbm_cycles,
                        "output_l2_cycles": output_l2_cycles,
                        "hbm_bytes": hbm_input + hbm_middle * 2 + hbm_output,
                        "l2_bytes": l2_input + l2_middle * 2 + l2_output,
                        "hbm_cycles": input_hbm_cycles + middle_hbm_cycles + output_hbm_cycles,
                        "l2_cycles": input_l2_cycles + middle_l2_cycles + output_l2_cycles,
                    }
                    events.append(event)
                    for name in totals:
                        if name in event:
                            totals[name] += event[name]
        return events, totals, q_tiles, kv_tiles

    @staticmethod
    def _schedule_tile_events(events, double_buffer):
        """Schedule the tile DAG with a duration-independent list order."""
        if not events:
            return 0
        array_count = max(1, int(events[0].get("array_count", 1)))
        vector_count = max(1, int(events[0].get("vector_count", 1)))
        buffer_count = 2 if double_buffer else 1
        stage_order = {
            "input": 0,
            "qk": 1,
            "softmax": 2,
            "middle": 3,
            "pv": 4,
            "output_update": 5,
            "output": 6,
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
        for index, event in enumerate(events):
            sequence = event.get("sequence", index)
            sequence_events = sequences[sequence]
            position = sequence_events.index(index)
            previous = sequence_events[position - 1] if position else None
            buffer_previous = (
                sequence_events[position - buffer_count]
                if position >= buffer_count
                else None
            )
            add_task(
                ("input", index),
                "input",
                event,
                [] if buffer_previous is None else [("output", buffer_previous)],
            )
            add_task(("qk", index), "qk", event, [("input", index)])
            softmax_deps = [("qk", index)]
            if previous is not None:
                softmax_deps.append(("softmax", previous))
            add_task(("softmax", index), "softmax", event, softmax_deps)
            add_task(("middle", index), "middle", event, [("softmax", index)])
            add_task(("pv", index), "pv", event, [("middle", index)])
            update_deps = [("pv", index)]
            if previous is not None:
                update_deps.append(("output_update", previous))
            add_task(
                ("output_update", index),
                "output_update",
                event,
                update_deps,
            )
            add_task(
                ("output", index),
                "output",
                event,
                [("output_update", index)],
            )

        def duration_for(stage, event, priority):
            if stage == "input":
                return max(
                    event.get("input_hbm_cycles", event.get("input", 0)),
                    event.get("input_l2_cycles", 0),
                )
            if stage == "qk":
                return event.get("qk", 0)
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
            if stage == "pv":
                return event.get("pv", 0)
            if stage == "output_update":
                return event.get("output_update", 0)
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
            finish = max(int(ready), state["memory"]) + duration
            state["memory"] = finish
            return finish

        def new_state():
            arrays = [0] * array_count
            vectors = [0] * vector_count
            heapq.heapify(arrays)
            heapq.heapify(vectors)
            return {"array": arrays, "vector": vectors, "memory": 0}

        # Build the list order with a variant-independent Softmax duration.
        # Reusing this order for actual durations prevents a faster kernel
        # from changing arbitration and producing a longer modeled makespan.
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
        return int(max(finish_times.values(), default=0))

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
        scheduled_cycles = self._schedule_tile_events(events, double_buffer)
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
            "scheduled_cycles": int(scheduled_cycles),
            "double_buffered": bool(double_buffer),
        }


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
