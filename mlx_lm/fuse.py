# Copyright © 2024 Apple Inc.

import argparse
from pathlib import Path

from mlx.utils import tree_flatten

from .gguf import convert_to_gguf
from .tuner.utils import fuse_adapters, load_adapters
from .utils import (
    dequantize_model,
    load,
    save,
    upload_to_hub,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse fine-tuned adapters into the base model."
    )
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        nargs="+",
        default=["adapters"],
        help="Path(s) to the trained adapter weights and config. Several "
        "adapters are fused into the base model in the order given.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dequantize",
        help="Generate a dequantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--export-gguf",
        help="Export model weights in GGUF format.",
        action="store_true",
    )
    parser.add_argument(
        "--gguf-path",
        help="Path to save the exported GGUF format model weights. Default is ggml-model-f16.gguf.",
        default="ggml-model-f16.gguf",
        type=str,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer/model loading.",
    )
    return parser.parse_args()


def main() -> None:
    print("Loading pretrained model")
    args = parse_arguments()

    model, tokenizer, config = load(
        args.model,
        return_config=True,
        trust_remote_code=args.trust_remote_code,
    )
    for adapter_path in args.adapter_path:
        model = load_adapters(model, adapter_path)
        model = fuse_adapters(model, dequantize=args.dequantize)

    if args.dequantize:
        print("Dequantizing model")
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)

    save_path = Path(args.save_path)
    save(
        save_path,
        args.model,
        model,
        tokenizer,
        config,
        donate_model=False,
    )

    if args.export_gguf:
        model_type = config["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        weights = dict(tree_flatten(model.parameters()))
        convert_to_gguf(save_path, weights, config, str(save_path / args.gguf_path))

    if args.upload_repo is not None:
        upload_to_hub(args.save_path, args.upload_repo)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.fuse...` directly is deprecated."
        " Use `mlx_lm.fuse...` or `python -m mlx_lm fuse ...` instead."
    )
    main()
