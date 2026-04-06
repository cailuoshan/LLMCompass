import argparse
from pathlib import Path

from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2

DEFAULT_CONFIG = Path("configs/XSAI.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the architecture JSON config file (default: configs/XSAI.json)",
    )
    return parser.parse_args()

def simulate_prefill_latency(system, bs, seq_len, heuristics):
    model = TransformerBlockInitComputationTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    _ = model(
        Tensor([bs, seq_len, 12288], data_type_dict["fp16"]),
    )
    latency_simulated = model.compile_and_simulate(system, heuristics)
    return latency_simulated

def simulate_decoding_latency(system, bs, seq_len, heuristics):
    # TODO: auto parse hyper_parameters from models on hugging face
    # Qwen3-0.6B
    # model_auto_regression = TransformerBlockAutoRegressionTP(
    #         d_model=1024,
    #         n_heads=16,
    #         device_count=1,
    #         data_type=data_type_dict["fp16"],
    #     )
    # _ = model_auto_regression(
    # 	Tensor([bs, 1, 1024], data_type_dict["fp16"]),
    # 	seq_len,
    # )
    # GPT-3
    model_auto_regression = TransformerBlockAutoRegressionTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    _ = model_auto_regression(
        Tensor([bs, 1, 12288], data_type_dict["fp16"]),
        seq_len,
    )
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, heuristics
    )
    return auto_regression_latency_simulated

def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path("configs") / config_path

    specs = read_architecture_template(str(config_path))
    system = template_to_system(specs)
    config_name = config_path.stem

    prefill_latency_s = simulate_prefill_latency(system, 16, 256, "heuristic-GPU")
    decode_latency_s = simulate_decoding_latency(system, 16, 256, "heuristic-GPU")
    # TODO: n_layers now is set to 1, we need to multiply the latency by n_layers to get the final latency

    print(f"Prefill latency for {config_name}: ", prefill_latency_s)
    print(f"Decode latency per token for {config_name}: ", decode_latency_s)

    print("=================================")
    compute_chiplet_area_mm2 = calc_compute_chiplet_area_mm2(specs)
    print(f"Compute chiplet area for {config_name} (mm^2): ", compute_chiplet_area_mm2)


if __name__ == "__main__":
    main()
