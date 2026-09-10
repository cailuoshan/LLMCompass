from llmcompass_utils import size
from hardware_model.device import Device
from software_model.design_point_recorder import (
    get_active_recorder,
    get_trial_latency_modifiers,
)
from software_model.operators import Operator
from software_model.search_protocol import (
    MappingCandidate,
    ProviderProtocolError,
    deduplicate_candidates,
    validate_ranked_ids,
)
from software_model.utils import Tensor, DataType
from math import ceil
import time
import statistics
import torch


@torch.compile
def gelu_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(input, approximate="tanh")


class GeLU(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:])
        self.computational_graph = self.ComputationalGraph(self.M, self.data_type)
        return input

    def roofline_model(self, pcb_module: Device):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        M = self.M
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = total_io_count / min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, data_type: DataType):
            self.M = M
            self.data_type = data_type

    def _candidate(self, vector_factor: int) -> MappingCandidate:
        return MappingCandidate(
            operator_name=self.recording_name or "GeLU",
            operator_type="GeLU",
            execution_kind="vector_mapping",
            strategy=None,
            graph={"M": self.M},
            mapping={"vector_factor": int(vector_factor)},
            precision={
                "input_word_size": self.computational_graph.data_type.word_size,
                "accuracy_class": "bounded_ulp",
            },
            layout={"contiguous": True, "axis": "last"},
            resource_requirements={"active_vector_count": int(vector_factor)},
        )

    def enumerate_transfer_candidates(
        self, pcb_module: Device, generation_mode="heuristic-GPU"
    ):
        if generation_mode not in (
            "heuristic-GPU",
            "heuristic-our-throughput",
            "heuristic-TPU",
            "exhaustive",
        ):
            raise ValueError("unsupported GeLU transfer candidate generation mode")
        vector_count = pcb_module.compute_module.core.vector_unit.vector_count
        factors = {1, vector_count}
        factor = 1
        while factor < vector_count:
            factors.add(factor)
            factor *= 2
        candidates = [
            self._candidate(vector_factor)
            for vector_factor in sorted(factors)
            if 0 < vector_factor <= vector_count
        ]
        candidates = deduplicate_candidates(candidates)
        if not candidates:
            raise ValueError("transfer candidate enumeration returned no GeLU mappings")
        return candidates

    def _simulate_candidate(self, pcb_module: Device, candidate: MappingCandidate):
        candidate = self.validate_candidate(candidate)
        vector_unit = pcb_module.compute_module.core.vector_unit
        vector_factor = int(candidate.mapping["vector_factor"])
        if vector_factor < 1 or vector_factor > vector_unit.vector_count:
            raise ProviderProtocolError("GeLU vector_factor exceeds hardware vector_count")
        data_type = self.computational_graph.data_type
        element_parallelism = (
            pcb_module.compute_module.core_count
            * vector_unit.vector_width
            * vector_factor
        )
        rounded_M = ceil(self.computational_graph.M / element_parallelism) * element_parallelism
        total_io_count = rounded_M * 2 * data_type.word_size
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )
        total_flop_count = rounded_M * (
            10 + vector_unit.flops_per_exp
        )
        active_fraction = vector_factor / vector_unit.vector_count
        compute_latency = (
            total_flop_count
            / (vector_unit.total_vector_flops_per_cycle * active_fraction)
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        latency = max(compute_latency, io_latency)
        cycle_count = ceil(latency * pcb_module.compute_module.clock_freq)
        return cycle_count, cycle_count / pcb_module.compute_module.clock_freq

    def _record_trial(self, pcb_module, candidate, cycle_count, latency, trial_sink):
        recorder = get_active_recorder()
        if recorder is not None and self.recording_name is not None:
            recorder.record_operator_mapping_trial(
                operator_name=self.recording_name,
                operator_type="GeLU",
                execution_kind=candidate.execution_kind,
                graph={"M": self.M},
                mapping=candidate.mapping,
                cycle_count=cycle_count,
                raw_local_latency_s=latency,
                candidate=candidate,
            )
        if trial_sink is not None:
            modifiers = get_trial_latency_modifiers()
            trial_sink(
                candidate.to_dict(),
                latency,
                cycle_count,
                modifiers.get("strategy"),
            )

    def evaluate_transfer_candidate(self, pcb_module, candidate, trial_sink=None):
        candidate = self.validate_candidate(candidate)
        cycle_count, latency = self._simulate_candidate(pcb_module, candidate)
        self._record_trial(pcb_module, candidate, cycle_count, latency, trial_sink)
        self.best_mapping = candidate.mapping
        self.best_cycle_count = cycle_count
        self.best_latency = latency
        self.execution_kind = candidate.execution_kind
        self.latency = latency
        return latency

    def _compile_and_simulate_transfer_learn(
        self,
        pcb_module,
        provider=None,
        stage=None,
        hardware=None,
        trial_sink=None,
        operator_name=None,
    ):
        if provider is None:
            raise ProviderProtocolError("transfer-learn requires a provider")
        candidates = self.enumerate_transfer_candidates(
            pcb_module, getattr(provider, "generation_mode", "heuristic-GPU")
        )
        records = [candidate.to_dict() for candidate in candidates]
        top_k = getattr(provider, "top_k", 1)
        ranked = provider.rank(
            operator_name or self.recording_name or "GeLU",
            stage,
            hardware or {},
            {},
            records,
            top_k,
        )
        selected = validate_ranked_ids(records, ranked, top_k)
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        best = None
        for candidate_id_value in selected:
            try:
                latency = self.evaluate_transfer_candidate(
                    pcb_module, by_id[candidate_id_value], trial_sink
                )
            except Exception as exc:
                raise RuntimeError(
                    "transfer candidate evaluation failed: stage={}, operator={}, candidate_id={}".format(
                        stage,
                        operator_name or self.recording_name,
                        candidate_id_value,
                    )
                ) from exc
            if best is None or latency < best[0]:
                best = latency, by_id[candidate_id_value]
        if best is None:
            raise ProviderProtocolError("provider selected no GeLU candidate")
        self.best_latency = best[0]
        self.best_mapping = best[1].mapping
        self.best_cycle_count = ceil(best[0] * pcb_module.compute_module.clock_freq)
        self.execution_kind = best[1].execution_kind
        self.latency = best[0]
        return self.latency

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str,
        provider=None,
        stage=None,
        hardware=None,
        trial_sink=None,
        operator_name=None,
    ):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        if compile_mode == "transfer-learn":
            return self._compile_and_simulate_transfer_learn(
                pcb_module,
                provider=provider,
                stage=stage,
                hardware=hardware,
                trial_sink=trial_sink,
                operator_name=operator_name,
            )
        vector_count = pcb_module.compute_module.core.vector_unit.vector_count
        candidate = self._candidate(vector_count)
        return self.evaluate_transfer_candidate(pcb_module, candidate, trial_sink)

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []
        for _ in range(3):
            _ = gelu_gpu(input)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = gelu_gpu(input)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
            latencies.append(end - start)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size_value = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size_value, size_value, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            gelu_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print(latencies)
        return avg_overhead
