"""Unified attention entry: model head semantics select MHA or GQA.

Only explicit QK/Softmax/AV is implemented. Fused kernels can be added behind
these implementations without changing Transformer block callers.
"""
from software_model.matmul import BatchedMatmul
from software_model.softmax import Softmax
from software_model.elementwise import Elementwise


class GQA:
    def __init__(self, layer_spec, device_count, data_type):
        self.nq = layer_spec.num_attention_heads // device_count
        self.nkv = layer_spec.num_key_value_heads // device_count
        self.group_size = self.nq // self.nkv
        self.head_dim = layer_spec.head_dim
        self.data_type = data_type

    def build(self, q, k, v, emit):
        b, query, _ = q.shape
        context = k.shape[1]
        d, g = self.head_dim, self.group_size
        # Materialize contiguous group-major layouts. Copies are charged;
        # cached K/V are never replicated to the number of Q heads.
        q = emit("Q_layout", Elementwise(self.data_type, "copy", [b, self.nkv, g * query, d]), q)
        k = emit("K_layout", Elementwise(self.data_type, "copy", [b, self.nkv, d, context]), k)
        v = emit("V_layout", Elementwise(self.data_type, "copy", [b, self.nkv, context, d]), v)
        scores = emit("Q_mul_K", BatchedMatmul(self.data_type), q, k)
        # Scaling and causal masking are identified in omitted_costs; like the
        # legacy model, QK/AV charge full rectangular explicit attention.
        probs = emit("A_softmax", Softmax(self.data_type), scores)
        heads = emit("A_mul_V", BatchedMatmul(self.data_type), probs, v)
        return emit("attention_output_layout", Elementwise(self.data_type, "copy", [b, query, self.nq*d]), heads)


class MHA(GQA):
    """One Q head per K/V head, using the same explicit primitives."""
    def __init__(self, layer_spec, device_count, data_type):
        super().__init__(layer_spec, device_count, data_type)
        if self.group_size != 1:
            raise ValueError("MHA requires equal Q and KV head counts")


def build_attention(layer_spec, device_count, data_type):
    nq, nkv = layer_spec.num_attention_heads, layer_spec.num_key_value_heads
    if nq < 1 or nkv < 1 or nq % nkv or nq % device_count or nkv % device_count:
        raise ValueError("invalid attention head grouping or tensor parallelism")
    return (MHA if nq == nkv else GQA)(layer_spec, device_count, data_type)
