"""Train a QLoRA adapter on prepared parliamentary speech records."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    data_path: Path
    output_dir: Path
    adapter_output: Path
    max_sequence_length: int
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    seed: int
    validation_fraction: float
    packing: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/training_data.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/training_run",
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/final_adapter",
    )
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack short records into full training sequences.",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> TrainingConfig:
    arguments = build_parser().parse_args(argv)
    if not arguments.data_path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {arguments.data_path}. "
            "Run scripts/prepare_data.py first or pass --data-path."
        )
    if not 0 < arguments.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if arguments.max_sequence_length < 1:
        raise ValueError("--max-sequence-length must be positive.")

    return TrainingConfig(
        model_name=arguments.model_name,
        data_path=arguments.data_path.resolve(),
        output_dir=arguments.output_dir.resolve(),
        adapter_output=arguments.adapter_output.resolve(),
        max_sequence_length=arguments.max_sequence_length,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
        packing=arguments.packing,
    )


def run_training(config: TrainingConfig) -> None:
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA-capable GPU.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.adapter_output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_sequence_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    dataset = load_dataset("json", data_files=str(config.data_path), split="train")
    if len(dataset) < 2:
        raise ValueError("Training requires at least two prepared records.")
    split = dataset.train_test_split(
        test_size=config.validation_fraction,
        seed=config.seed,
    )
    supports_bfloat16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=config.max_sequence_length,
        packing=config.packing,
        args=SFTConfig(
            output_dir=str(config.output_dir),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            bf16=supports_bfloat16,
            fp16=not supports_bfloat16,
            optim="adamw_8bit",
            seed=config.seed,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(str(config.adapter_output))
    tokenizer.save_pretrained(str(config.adapter_output))
    print(f"Adapter saved to {config.adapter_output}")


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    print(
        "Training configuration: "
        f"model={config.model_name}, data={config.data_path}, "
        f"max_sequence_length={config.max_sequence_length}, epochs={config.epochs}, "
        f"effective_batch_size={config.batch_size * config.gradient_accumulation_steps}"
    )
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
