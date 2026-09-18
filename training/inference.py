"""Generate an explicitly synthetic completion with the local LoRA adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_NOTICE = (
    "SYNTHETIC OUTPUT: This text is model-generated and is not an authentic "
    "statement or quotation."
)


@dataclass(frozen=True)
class InferenceConfig:
    adapter_path: Path
    prompt: str
    max_sequence_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Completion prompt to test.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/final_adapter",
    )
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_config(argv: Sequence[str] | None = None) -> InferenceConfig:
    arguments = build_parser().parse_args(argv)
    if not arguments.adapter_path.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found at {arguments.adapter_path}. "
            "Pass --adapter-path after downloading or training the adapter."
        )
    if arguments.max_new_tokens < 1 or arguments.max_sequence_length < 1:
        raise ValueError("Token limits must be positive.")
    if not 0 < arguments.top_p <= 1:
        raise ValueError("--top-p must be greater than 0 and at most 1.")
    if arguments.temperature <= 0:
        raise ValueError("--temperature must be positive.")

    return InferenceConfig(
        adapter_path=arguments.adapter_path.resolve(),
        prompt=arguments.prompt.strip(),
        max_sequence_length=arguments.max_sequence_length,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        repetition_penalty=arguments.repetition_penalty,
        seed=arguments.seed,
    )


def generate_completion(config: InferenceConfig) -> str:
    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit adapter inference requires a CUDA-capable GPU.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(config.adapter_path),
        max_seq_length=config.max_sequence_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    torch.manual_seed(config.seed)
    inputs = tokenizer(config.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        do_sample=True,
    )
    generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    print(SYNTHETIC_NOTICE)
    print(generate_completion(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
