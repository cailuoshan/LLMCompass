"""RMS normalization: square/reduce, reciprocal root, scale and weight."""
from software_model.elementwise import Elementwise


class RMSNorm(Elementwise):
    def __init__(self, data_type, eps=1e-6):
        super().__init__(data_type, "rmsnorm")
        self.eps = eps

    def __call__(self, input):
        output = super().__call__(input)
        rows = input.size // input.shape[-1]
        self.flop_count = 4 * input.size + 3 * rows
        self.io_count += input.shape[-1] * self.data_type.word_size
        return output
