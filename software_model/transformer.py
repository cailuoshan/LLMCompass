"""Configuration-driven dense Transformer graphs and serial layer aggregation."""
import math
from dataclasses import dataclass, asdict
from enum import Enum

from software_model.attention import build_attention
from software_model.matmul import Matmul, BatchedMatmul
from software_model.layernorm import LayerNorm
from software_model.softmax import Softmax
from software_model.gelu import GeLU
from software_model.rmsnorm import RMSNorm
from software_model.elementwise import Elementwise
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.utils import Tensor, data_type_dict
from software_model.workload import LayerSpec, WorkloadSpec, fingerprint, EVALUATOR_VERSION
from software_model.search_protocol import operator_problem, stable_id
from software_model.design_point_recorder import get_active_recorder


def _hardware_signature(value):
    """Serialize hardware parameters, including direct callers without a template."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("hardware parameters must not contain NaN")
        # Existing interconnect models use infinity for unlimited on-package bandwidth.
        return "unbounded" if value > 0 else "negative_infinity"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _hardware_signature(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_hardware_signature(v) for v in value]
    return {k: _hardware_signature(v) for k, v in vars(value).items() if not k.startswith("_")}


@dataclass
class StageResult:
    latency_s: float
    selected_software_config: dict

    def to_dict(self):
        return asdict(self)


class TransformerBlock:
    def __init__(self, layer_spec, workload_spec, stage, device_count, software_config=None):
        self.layer_spec, self.workload_spec = layer_spec, workload_spec
        self.stage, self.device_count = stage, device_count
        if stage not in ("prefill", "decode"):
            raise ValueError("stage must be prefill or decode")
        for field in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
            if getattr(layer_spec, field) % device_count:
                raise ValueError(f"TP must divide {field}")
        self.data_type = data_type_dict[workload_spec.dtype]
        self.policy = {"qkv_projection": "separate", "gate_up_projection": "separate", **(software_config or {})}
        if any(k not in ("qkv_projection", "gate_up_projection") or v not in ("separate", "fused") for k,v in self.policy.items()):
            raise ValueError("unsupported software policy")
        if layer_spec.ffn_type == "gelu":
            self.policy["gate_up_projection"] = "separate"
        self.nodes, self.weights, self.externals = [], [], {}
        self.tensor_names, self.tensor_sizes = {}, {}
        self._build()

    def register(self, tensor, prefix="tensor", external=False):
        key = id(tensor)
        if key not in self.tensor_names:
            self.tensor_names[key] = f"{prefix}_{len(self.tensor_names)}"
            self.tensor_sizes[self.tensor_names[key]] = tensor.size * tensor.data_type.word_size
        name = self.tensor_names[key]
        if external:
            self.externals[name] = tensor
        return name

    def emit(self, role, op, *inputs):
        op.recording_name = role
        out = op(*inputs)
        # Some existing operators return their input Tensor; graph values still
        # need distinct identities because explicit kernels materialize outputs.
        if any(out is x for x in inputs):
            out = Tensor(out.shape, out.data_type)
        input_names = [self.register(x) for x in inputs]
        output_name = self.register(out, role)
        self.nodes.append({"role": role, "op": op, "inputs": input_names, "output": output_name,
                           "input_shapes": [list(x.shape) for x in inputs], "output_shape": list(out.shape),
                           "tensors": (*inputs, out)})
        return out

    def project(self, role, x, width):
        w = Tensor([x.shape[-1], width], self.data_type)
        self.weights.append(w)
        self.register(w, "weight", external=True)
        return self.emit(role, Matmul(self.data_type), x, w)

    def norm(self, role, x):
        spec = self.layer_spec
        op = LayerNorm(self.data_type) if spec.norm_type == "layernorm" else RMSNorm(self.data_type, spec.norm_eps)
        return self.emit(role, op, x)

    def _build(self):
        l, w, p = self.layer_spec, self.workload_spec, self.device_count
        b = w.batch_size
        qlen = w.input_seq_len if self.stage == "prefill" else 1
        kvlen = w.input_seq_len if self.stage == "prefill" else w.decode_context_len + 1
        qwidth, kvwidth, intermediate = l.num_attention_heads*l.head_dim//p, l.num_key_value_heads*l.head_dim//p, l.intermediate_size//p
        x = Tensor([b, qlen, l.hidden_size], self.data_type)
        self.input = x
        self.register(x, "input")
        xn = self.norm("layer_norm0", x)
        if self.policy["qkv_projection"] == "fused":
            packed = self.project("QKV_proj", xn, qwidth + 2*kvwidth)
            # Explicit unpack copies: no free noncontiguous view assumption.
            q = self.emit("Q_unpack", Elementwise(self.data_type, "copy", [b,qlen,qwidth], read_elements=b*qlen*qwidth), packed)
            k = self.emit("K_unpack", Elementwise(self.data_type, "copy", [b,qlen,kvwidth], read_elements=b*qlen*kvwidth), packed)
            v = self.emit("V_unpack", Elementwise(self.data_type, "copy", [b,qlen,kvwidth], read_elements=b*qlen*kvwidth), packed)
        else:
            q, k, v = [self.project(role, xn, width) for role,width in (("Q_proj",qwidth),("K_proj",kvwidth),("V_proj",kvwidth))]
        if l.qk_norm:
            q = self.emit("Q_head_layout", Elementwise(self.data_type, "copy", [b,qlen,l.num_attention_heads//p,l.head_dim]), q)
            k = self.emit("K_head_layout", Elementwise(self.data_type, "copy", [b,qlen,l.num_key_value_heads//p,l.head_dim]), k)
            q = self.emit("Q_norm", RMSNorm(self.data_type, l.norm_eps), q)
            k = self.emit("K_norm", RMSNorm(self.data_type, l.norm_eps), k)
            q = self.emit("Q_flatten", Elementwise(self.data_type, "copy", [b,qlen,qwidth]), q)
            k = self.emit("K_flatten", Elementwise(self.data_type, "copy", [b,qlen,kvwidth]), k)
        # Writes charge only new tokens. The output refers to resident cache,
        # whose total capacity is counted separately for every physical layer.
        k = self.emit("KV_write_K", Elementwise(self.data_type, "copy", [b,kvlen,kvwidth], write_elements=b*qlen*kvwidth), k)
        v = self.emit("KV_write_V", Elementwise(self.data_type, "copy", [b,kvlen,kvwidth], write_elements=b*qlen*kvwidth), v)
        self.register(k, external=True)
        self.register(v, external=True)
        self.attention = build_attention(l, p, self.data_type)
        out = self.attention.build(q, k, v, self.emit)
        out = self.project("H_matmul0", out, l.hidden_size)
        if p > 1:
            out = self.emit("allreduce_mha", AllReduceMultiPCB(self.data_type), out)
        x = self.emit("residual_attention", Elementwise(self.data_type), x, out)
        xn = self.norm("layer_norm1", x)
        if l.ffn_type == "gelu":
            hidden = self.project("H_matmul1", xn, intermediate)
            hidden = self.emit("H_gelu", GeLU(self.data_type), hidden)
        else:
            if self.policy["gate_up_projection"] == "fused":
                packed = self.project("gate_up_proj", xn, 2*intermediate)
                gate = self.emit("gate_unpack", Elementwise(self.data_type, "copy", [b,qlen,intermediate], read_elements=b*qlen*intermediate), packed)
                up = self.emit("up_unpack", Elementwise(self.data_type, "copy", [b,qlen,intermediate], read_elements=b*qlen*intermediate), packed)
            else:
                gate = self.project("gate_proj", xn, intermediate)
                up = self.project("up_proj", xn, intermediate)
            hidden = self.emit("swiglu", Elementwise(self.data_type, "swiglu"), gate, up)
        out = self.project("H_matmul2", hidden, l.hidden_size)
        if p > 1:
            out = self.emit("allreduce_ffn", AllReduceMultiPCB(self.data_type), out)
        self.output = self.emit("residual_ffn", Elementwise(self.data_type), x, out)
        self.kv_cache_bytes = 2*b*kvlen*kvwidth*self.data_type.word_size
        # Norm weights are replicated; Q/K head norm uses one D-vector each.
        norm_weights = 2*l.hidden_size*(2 if l.norm_type == "layernorm" else 1)
        bias_weights = qwidth+2*kvwidth+l.hidden_size if l.attention_bias else 0
        self.resident_weight_bytes = (sum(t.size for t in self.weights) + norm_weights + bias_weights + (2*l.head_dim if l.qk_norm else 0))*self.data_type.word_size

    def memory_summary(self):
        last_use = {}
        for i,n in enumerate(self.nodes):
            for name in n["inputs"]:
                last_use[name] = i
        live = {self.tensor_names[id(self.input)]}
        peak = 0
        for i,n in enumerate(self.nodes):
            if n["output"] not in self.externals:
                live.add(n["output"])
            peak = max(peak, sum(self.tensor_sizes[name] for name in live))
            live = {name for name in live if last_use.get(name, i) > i}
        communication = self.input.size*self.data_type.word_size if self.device_count > 1 else 0
        return {"resident_weight_bytes": self.resident_weight_bytes, "kv_cache_bytes": self.kv_cache_bytes,
                "peak_live_activation_and_workspace_bytes": peak, "communication_buffers_bytes": communication,
                "peak_bytes": self.resident_weight_bytes+self.kv_cache_bytes+peak+communication}

    def compile_and_simulate(self, system, compile_mode, provider=None, stage=None, hardware=None, trial_sink=None, evaluation_cache=None):
        if get_active_recorder() is not None:
            raise ValueError("configuration-driven Transformer requires structured trial_sink; legacy wide recorder is unsupported")
        cache = evaluation_cache if evaluation_cache is not None else {}
        total_latency = 0.0
        hardware_spec = _hardware_signature(system)
        for n in self.nodes:
            op, role = n["op"], n["role"]
            semantic = {"operator_type": type(op).__name__, "input_shapes": n["input_shapes"], "output_shape": n["output_shape"],
                        "dtype": self.data_type.name, "stage": self.stage, "compile_mode": compile_mode,
                        "device_count": self.device_count, "evaluator_version": EVALUATOR_VERSION}
            if isinstance(op, BatchedMatmul):
                semantic.update(batch=op.bs, group_size=self.attention.group_size, kv_heads=self.attention.nkv,
                                kv_sharing="grouped", layout="materialized_group_major")
            if isinstance(op, Elementwise):
                semantic.update(kind=op.kind, io_bytes=op.io_count, flops=op.flop_count)
            if isinstance(op, RMSNorm):
                semantic["eps"] = op.eps
            key = stable_id([role, semantic, hardware_spec, hardware or {}])
            if key not in cache:
                with operator_problem(semantic):
                    if isinstance(op, AllReduceMultiPCB):
                        latency = op.simulate(system.interconnect)
                        overhead = 0
                    elif compile_mode == "roofline":
                        latency = op.roofline_model(system.device)
                        overhead = self._overhead(op, system.device)
                    elif isinstance(op, GeLU):
                        latency = op.compile_and_simulate(system.device, compile_mode)
                        overhead = self._overhead(op, system.device)
                    else:
                        latency = op.compile_and_simulate(system.device, compile_mode, provider=provider, stage=self.stage, hardware=hardware, trial_sink=trial_sink)
                        overhead = self._overhead(op, system.device)
                if not math.isfinite(float(latency)) or latency < 0:
                    raise ValueError(f"invalid {role} latency: {latency}")
                cache[key] = float(latency + overhead)
            total_latency += cache[key]
        self.latency = total_latency
        return {"latency_s": self.latency}

    @staticmethod
    def _overhead(op, device):
        kind = "matmul" if isinstance(op,(Matmul,BatchedMatmul)) else "softmax" if isinstance(op,Softmax) else "layernorm" if isinstance(op,(LayerNorm,RMSNorm)) else "gelu"
        return float(getattr(device.compute_module.overhead, kind))


class TransformerModel:
    def __init__(self, model_spec, workload_spec, stage, device_count, software_config=None):
        self.model_spec, self.workload_spec = model_spec, workload_spec
        self.stage, self.device_count = stage, device_count
        self.software_config = software_config or {"qkv_projection":"separate", "gate_up_projection":"separate"}
        # Keep one graph per unique layer specification.  ``layer_count`` is
        # applied to latency and persistent memory after that graph has been
        # evaluated once, so homogeneous models never repeat mapping searches.
        self.block_groups = []
        unique = {}
        for spec in model_spec.layers:
            key = fingerprint(spec.to_dict())
            if key not in unique:
                group = {
                    "block": TransformerBlock(
                        spec, workload_spec, stage, device_count, self.software_config
                    ),
                    "layer_count": 0,
                }
                unique[key] = group
                self.block_groups.append(group)
            unique[key]["layer_count"] += 1
        # Compatibility for callers that inspect the constructed unique blocks.
        self.blocks = tuple(group["block"] for group in self.block_groups)

    def memory_summary(self):
        memories = [
            (group["block"].memory_summary(), group["layer_count"])
            for group in self.block_groups
        ]
        result = {
            key: sum(memory[key] * count for memory, count in memories)
            for key in ("resident_weight_bytes", "kv_cache_bytes")
        }
        result.update({
            key: max(memory[key] for memory, _ in memories)
            for key in (
                "peak_live_activation_and_workspace_bytes",
                "communication_buffers_bytes",
            )
        })
        result["peak_bytes"] = sum(result.values())
        return result

    def compile_and_simulate(self, system, compile_mode, provider=None, stage=None, hardware=None, trial_sink=None, evaluation_cache=None):
        cache = evaluation_cache if evaluation_cache is not None else {}
        total_latency = 0.0
        for group in self.block_groups:
            block = group["block"]
            result = block.compile_and_simulate(system,compile_mode,provider,self.stage,hardware,trial_sink,cache)
            total_latency += result["latency_s"] * group["layer_count"]
        return StageResult(total_latency, self.software_config)



class TransformerBlockInitComputationTP:
    """Compatibility adapter for existing single-block microbench callers."""
    stage = "prefill"
    def __init__(self, d_model, n_heads, device_count, data_type):
        if d_model % n_heads:
            raise ValueError("legacy n_heads must divide d_model")
        self.d_model, self.n_heads, self.device_count, self.data_type = d_model,n_heads,device_count,data_type
        self.spec = LayerSpec(d_model,n_heads,n_heads,d_model//n_heads,4*d_model,"layernorm","gelu","learned_absolute")
    def set_recording_names(self):
        pass  # Stable roles are assigned during graph construction.
    def __call__(self, x, seq_len=None):
        if x.shape[-1] != self.d_model or x.data_type != self.data_type:
            raise ValueError("block input does not match model")
        workload = WorkloadSpec(x.shape[0],x.shape[1],seq_len if seq_len is not None else x.shape[1],self.data_type.name)
        self.block = TransformerBlock(self.spec,workload,self.stage,self.device_count)
        return self.block.output
    def compile_and_simulate(self, system, compile_mode, **kwargs):
        self.latency = self.block.compile_and_simulate(system,compile_mode,**kwargs)["latency_s"]
        return self.latency
    def roofline_model(self, system):
        return self.compile_and_simulate(system,"roofline")


class TransformerBlockAutoRegressionTP(TransformerBlockInitComputationTP):
    stage = "decode"
