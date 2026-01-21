# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SalBench (Saliency Benchmark) is a benchmark for evaluating perceptual capabilities of Vision-Language Models. It tests LVLM ability to detect visually salient features like color, orientation, and size anomalies in images.

## Package Manager

This project uses **uv** as the package manager. All commands should be run via `uv run` or `uv pip`.

## Common Commands

### Setup
```bash
uv pip install -r requirements.txt
./scripts/download_font.sh  # Downloads SVG fonts required for orientation/size data generation
```

### Data Generation

**Pretraining data** (outputs tar files with images + captions):
```bash
uv run ./scripts/pretrain_data_generation.sh
```

**SFT data** (outputs folders with images + conversation JSON):
```bash
uv run ./scripts/sft_data_generation.sh
```

**Individual generators** (all support `--main_path`, `--total_images`, `--images_per_tar`/`--images_per_folder`, `--tokenizer_model`):
```bash
uv run python ./data_generation/pretrain/gen_color.py --total_images 1000
uv run python ./data_generation/pretrain/gen_orientation.py --total_images 1000
uv run python ./data_generation/pretrain/gen_size.py --total_images 1000
```

## Architecture

### Data Generation Pipeline

Two parallel pipelines exist under `data_generation/`:

- **pretrain/**: Generates image-caption pairs packed into tar files (WebDataset format). Each tar contains `.npy` (images) and `.txt` (captions) files. Uses `<image>caption<|endoftext|>` format.

- **sft/**: Generates images with multi-turn conversation JSON for supervised fine-tuning. Creates folder structure with `images/` and `json/` subdirectories. Conversations follow `{"from": "human/gpt", "value": ...}` format.

### Saliency Feature Types

Each pipeline has three generators for different visual saliency features:

1. **gen_color.py**: Creates images with one shape of a different color among distractors (uses PIL shapes)
2. **gen_orientation.py**: Creates images with one rotated icon among uniformly-oriented icons (requires SVG fonts from `Fonts/svgs/regular/`)
3. **gen_size.py**: Creates images with one larger/smaller icon among uniform icons (requires SVG fonts from `Fonts/svgs/`)

### Evaluation Framework

The `evaluation/` module runs the SalBench benchmark against vision-language models via Ollama or OpenRouter APIs. It uses the official P3/O3 benchmark datasets from HuggingFace (`salbench-vlm/salbench`).

**Task Definitions (from paper):**
- **Detection (D)**: Classify which feature type(s) differ in the image
- **Referring (R)**: Given bbox coordinates in prompt, classify the feature type
- **Visual Referring (VR)**: Given red-box overlay image, classify the feature type

**Dataset Splits:**
- **Synthetic (P3)**: 3 feature classes - Orientation, Color, Size
- **Natural (O3)**: 7 feature classes - Orientation, Color, Focus, Shape, Size, Location, Pattern

**Run evaluation:**
```bash
# Local Ollama - uses OLLAMA_HOST env var or localhost:11434
uv run python scripts/run_evaluation.py --backend ollama --model llava:7b --splits synthetic

# Local Ollama - explicit base URL
uv run python scripts/run_evaluation.py --backend ollama --base-url http://192.168.1.100:11434 \
    --model llava:7b --splits synthetic

# OpenRouter - both splits (requires OPENROUTER_API_KEY env var)
uv run python scripts/run_evaluation.py --backend openrouter --model qwen/qwen2-vl-72b-instruct \
    --splits synthetic,natural

# Quick test
uv run python scripts/run_evaluation.py --samples 10 --tasks D --splits synthetic
```

**Environment variables:**
- `OLLAMA_HOST`: Base URL for Ollama backend (default: `http://localhost:11434`)
- `OPENROUTER_API_KEY`: API key for OpenRouter backend

**Key options:**
- `--backend`: `ollama` or `openrouter`
- `--base-url`: Override API base URL
- `--model`: Model name (e.g., `llava:7b`, `qwen/qwen2-vl-72b-instruct`)
- `--dataset`: HuggingFace dataset name (default: `salbench-vlm/salbench`)
- `--splits`: Comma-separated: `synthetic,natural`
- `--tasks`: `D` (Detection), `R` (Referring), `VR` (Visual Referring)
- `--shots`: Few-shot examples: `0`, `3`, or `5`
- `--samples`: Samples per split (default: all)
- `--timeout`: Request timeout in seconds (default: 120)
- `--verbose`, `-v`: Enable debug logging

**Output features:**
- Rich colorized console output with panels and tables
- Elapsed time tracking (displayed in seconds)
- Token usage tracking (input/output tokens from API responses)
- Cost estimation for OpenRouter based on model pricing
- Color-coded F1 scores: green (>=70%), yellow (>=40%), red (<40%)
- Results saved to `./results/<model>_<timestamp>/`

**Evaluation modules:**
- `config.py`: Configuration dataclass with HuggingFace dataset settings
- `vision_client.py`: Unified API client (Ollama + OpenRouter), handles PIL Images
- `data_loader.py`: Load benchmark data from HuggingFace datasets
- `prompts.py`: Exact prompts from the SalBench paper
- `response_parser.py`: Multi-label response parsing
- `metrics.py`: Multi-label F1 and exact match computation
- `evaluator.py`: Main orchestrator with usage tracking and rich output

### Key Dependencies

- **svglib/reportlab/pycairo**: SVG rendering for orientation and size generators
- **PIL**: Shape drawing for color generator
- **transformers**: Tokenizer for computing sample token lengths (batching by max_tokens_per_sample)
- **aiohttp**: Async HTTP client for API requests
- **rich/rich-click**: Colorized console output and CLI help
