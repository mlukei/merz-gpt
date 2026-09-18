# German Parliamentary Speech Style LoRA

A technical case study in parameter-efficient fine-tuning. The
project applies 4-bit QLoRA with Unsloth and TRL to Meta Llama 3.1 8B Instruct,
using publicly accessible German Bundestag plenary-protocol data.

## Architecture

```text
Bundestag DIP API
  -> bundestag-protocol-extractor CSV exports
  -> exact speaker and XML-quality filtering
  -> deterministic text cleaning and JSONL export
  -> Llama 3.1 8B Instruct loaded in 4 bit
  -> LoRA supervised fine-tuning with Unsloth + TRL
  -> adapter-only local artifact
```

The adapter modifies attention and feed-forward projection layers while the
quantized base-model weights remain frozen. Training uses plain text completion.

## Data

The local corpus was derived from the German Bundestag's official DIP plenary
protocols with `bundestag-protocol-extractor` 0.3.3. The reproducible default
pipeline selects exact person records for Friedrich Merz.

Download and prepare the data from the repository root:

```bash
python scripts/prepare_data.py --download
```

The extractor includes the Bundestag's public DIP API key. For a separately
issued key, set `BUNDESTAG_API_KEY`

## QLoRA configuration

| Setting | Value |
| --- | --- |
| Training checkpoint | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| Adapter base-model metadata | `unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit` |
| Quantization | 4-bit base-model loading |
| Maximum sequence length | 1,024 tokens |
| LoRA rank / alpha / dropout | 16 / 16 / 0 |
| Target modules | `q`, `k`, `v`, `o`, `gate`, `up`, `down` projections |
| Gradient checkpointing | Unsloth |
| Train/evaluation split | 90% / 10%, seed 42 |
| Epochs | 3 |
| Device batch / accumulation | 1 / 8 (effective batch size 8) |
| Optimizer | 8-bit AdamW |
| Learning rate / schedule | 2e-4 / cosine, 3% warmup |
| Packing | Enabled |

These are experiment settings, not generally optimal defaults.

## Hardware and installation

Create a clean environment, install the appropriate PyTorch build from the
[official selector](https://pytorch.org/get-started/locally/), then install the
verified project versions:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build matching the local CUDA driver first.
python -m pip install -r requirements.txt
```

## Training

Prepare `data/processed/training_data.jsonl`, then run:

```bash
python training/train_lora.py \
  --data-path /path/to/training_data.jsonl \
  --adapter-output /path/to/final_adapter \
  --max-sequence-length 1024
```

## Inference

Inference is completion-style because training records are plain speech text.

```bash
python training/inference.py \
  --prompt "This synthetic parliamentary-format example concerns the meeting schedule."
```

## Repository structure

```text
data/examples/                 Synthetic schema fixture only
notebooks/data_preparation.ipynb
scripts/prepare_data.py        Canonical deterministic data pipeline
training/train_lora.py         Configurable QLoRA training CLI
training/inference.py          Explicitly synthetic completion CLI
artifacts/final_adapter/       Local final adapter; ignored by Git
requirements*.txt              Runtime and development dependencies
```
