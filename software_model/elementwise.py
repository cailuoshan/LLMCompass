"""Fixed analytical vector/memory costs; these are not mapping search knobs."""
from software_model.operators import Operator
from software_model.utils import Tensor


class Elementwise(Operator):
    def __init__(self, data_type, kind="add", output_shape=None, read_elements=None, write_elements=None):
        super().__init__(0, 0, 0, 0, data_type)
        self.kind = kind
        self.output_shape = output_shape
        self.read_elements = read_elements
        self.write_elements = write_elements

    def __call__(self, *inputs):
        if any(x.data_type != self.data_type for x in inputs):
            raise ValueError("elementwise dtype mismatch")
        if self.kind in ("add", "swiglu") and (len(inputs) != 2 or inputs[0].shape != inputs[1].shape):
            raise ValueError(f"{self.kind} requires two equal shapes")
        output = Tensor(self.output_shape or inputs[0].shape, self.data_type)
        self.elements = output.size
        self.io_count = ((sum(x.size for x in inputs) if self.read_elements is None else self.read_elements)
                         + (output.size if self.write_elements is None else self.write_elements)) * self.data_type.word_size
        self.flop_count = output.size if self.kind == "add" else 0
        return output

    def compile_and_simulate(self, device, compile_mode=None, **kwargs):
        flops = self.flop_count
        if self.kind == "swiglu":
            flops = self.elements * (4 + device.compute_module.core.vector_unit.flops_per_exp)
        bandwidth = min(device.io_module.bandwidth, device.compute_module.l2_bandwidth_per_cycle * device.compute_module.clock_freq)
        self.latency = max(self.io_count / bandwidth, flops / device.compute_module.total_vector_flops)
        return self.latency

    roofline_model = compile_and_simulate
