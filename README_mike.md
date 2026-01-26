# SalBench - Saliency Benchmark for Vision-Language Models

This repository contains tools for benchmarking Vision-Language Models (VLMs) on their ability to detect **visually salient features** - things that "pop out" like color, orientation, and size anomalies in images.

## What We Built

### 1. Data Generation Pipeline

Two parallel pipelines for generating training data under `data_generation/`:

| Pipeline | Output Format | Use Case |
|----------|--------------|----------|
| **pretrain/** | WebDataset tar files (`.npy` + `.txt`) | Image-caption pretraining |
| **sft/** | Folders with images + conversation JSON | Supervised fine-tuning |

Three saliency feature types are generated:
- **Color**: One shape with a different color among distractors
- **Orientation**: One rotated icon among uniformly-oriented icons (requires SVG fonts)
- **Size**: One larger/smaller icon among uniform icons (requires SVG fonts)

### 2. Evaluation Framework

A complete framework under `evaluation/` for benchmarking VLMs against the official SalBench dataset from HuggingFace (`salbench-vlm/salbench`).

**Features:**
- Support for **Ollama** (local) and **OpenRouter** (API) backends
- Three task types matching the SalBench paper:
  - **Detection (D)**: Identify which feature type(s) differ in the image
  - **Referring (R)**: Given bbox coordinates in prompt, classify the feature type
  - **Visual Referring (VR)**: Given red-box overlay image, classify the feature type
- Multi-label classification with F1/precision/recall/exact-match metrics
- Rich colorized console output with progress bars
- Token usage tracking and cost estimation (for OpenRouter)
- Results saved to JSON and CSV for easy analysis

### 3. Evaluation Results

We have run evaluations on several models. Results on the **synthetic (P3)** split:

| Model | Backend | Detection | Referring | VisualRef | Cost | Samples |
|-------|---------|-----------|-----------|-----------|------|---------|
| **openai/gpt-4o** | OpenRouter | 86.3% | 86.7% | 76.3% | $16.25 | 2589/task |
| **qwen/qwen2.5-vl-72b-instruct** | OpenRouter | 67.0% | 88.7% | 78.6% | $1.70 | 2589/task |
| **anthropic/claude-3.5-sonnet** | OpenRouter | 65.0%* | 70.0%* | 50.0%* | $0.14 | 10/task |

*Claude results are from a small sample run (10 samples) - not statistically significant.

**Key Finding**: Qwen2-VL-72B offers excellent value - comparable performance to GPT-4o on Referring/VisualRef tasks at ~10x lower cost.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# (Optional) Download SVG fonts for data generation
./scripts/download_font.sh
```

### Environment Variables

```bash
# For Ollama (optional - defaults to localhost:11434)
export OLLAMA_HOST=http://your-ollama-server:11434

# For OpenRouter (required for API-based evaluation)
export OPENROUTER_API_KEY=your-api-key
```

---

## Running Evaluations

### Quick Test (10 samples)

```bash
# Test with OpenRouter
uv run python scripts/run_evaluation.py \
    --backend openrouter \
    --model qwen/qwen2.5-vl-7b-instruct \
    --splits synthetic \
    --tasks D \
    --samples 10
```

### Local Ollama Evaluation

```bash
# Using default localhost
uv run python scripts/run_evaluation.py \
    --backend ollama \
    --model llava:7b \
    --splits synthetic \
    --tasks D,R,VR

# Using remote Ollama server
uv run python scripts/run_evaluation.py \
    --backend ollama \
    --base-url http://192.168.25.202:11434 \
    --model llava:7b \
    --splits synthetic \
    --tasks D,R,VR
```

### OpenRouter API Evaluation

```bash
# Qwen2-VL-72B (best value)
uv run python scripts/run_evaluation.py \
    --backend openrouter \
    --model qwen/qwen2.5-vl-72b-instruct \
    --splits synthetic \
    --tasks D,R,VR

# GPT-4o (best performance)
uv run python scripts/run_evaluation.py \
    --backend openrouter \
    --model openai/gpt-4o \
    --splits synthetic \
    --tasks D,R,VR

# Both synthetic and natural splits
uv run python scripts/run_evaluation.py \
    --backend openrouter \
    --model qwen/qwen2.5-vl-72b-instruct \
    --splits synthetic,natural \
    --tasks D,R,VR
```

### Using Batch Files (Windows)

Pre-configured batch files are available:

```batch
runeval_local.bat              # Ollama on remote server (llava:7b)
runeval_openrouter.bat         # OpenRouter with GPT-4o (full run)
runeval_openrouter_10.bat      # OpenRouter with Claude (10 samples)
runeval_openrouter_gpt4o.bat   # OpenRouter with GPT-4o (full run)
```

### CLI Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | `ollama` or `openrouter` | `ollama` |
| `--base-url` | Override API base URL | auto |
| `--api-key` | OpenRouter API key | `$OPENROUTER_API_KEY` |
| `--model` | Model name | `llava:7b` |
| `--dataset` | HuggingFace dataset | `salbench-vlm/salbench` |
| `--splits` | Comma-separated: `synthetic`, `natural` | `synthetic` |
| `--tasks` | Comma-separated: `D`, `R`, `VR` | `D,R,VR` |
| `--shots` | Few-shot examples: `0`, `3`, `5` | `0` |
| `--samples` | Samples per split (null = all) | all |
| `--output` | Results directory | `./results` |
| `--timeout` | Request timeout (seconds) | `120` |
| `--verbose`, `-v` | Debug logging | off |

---

## Data Generation

### Generate Pretraining Data

```bash
# Run all generators
uv run ./scripts/pretrain_data_generation.sh

# Or run individual generators
uv run python ./data_generation/pretrain/gen_color.py --total_images 1000
uv run python ./data_generation/pretrain/gen_orientation.py --total_images 1000
uv run python ./data_generation/pretrain/gen_size.py --total_images 1000
```

Output: `./data/saliency/pretrain/` with tar files containing `.npy` (images) and `.txt` (captions)

### Generate SFT Data

```bash
# Run all generators
uv run ./scripts/sft_data_generation.sh

# Or run individual generators
uv run python ./data_generation/sft/gen_color.py --total_images 1000
```

Output: `./data/saliency/sft/` with `images/` and `json/` subdirectories

---

## Output Files

Results are saved to `./results/<model>_<timestamp>/`:

| File | Contents |
|------|----------|
| `metrics.json` | Full report: config, usage stats, leaderboard row, detailed metrics |
| `raw_responses.json` | Individual sample results with predictions and scores |
| `leaderboard_row.csv` | Single CSV row matching paper's Table 1 format |

### Example metrics.json

```json
{
  "config": {
    "model": "openai/gpt-4o",
    "backend": "openrouter",
    "splits": ["synthetic"],
    "tasks": ["D", "R", "VR"],
    "num_shots": 0
  },
  "usage": {
    "elapsed_seconds": 8679.6,
    "total_requests": 7767,
    "input_tokens": 6470343,
    "output_tokens": 7781,
    "estimated_cost_usd": 16.25
  },
  "leaderboard_row": {
    "Model": "openai/gpt-4o",
    "Shot": 0,
    "Detection_SYN": 86.3,
    "Referring_SYN": 86.7,
    "VisualRef_SYN": 76.3
  }
}
```

---

## Project Structure

```
salbench/
├── evaluation/              # Core evaluation framework
│   ├── config.py           # Configuration dataclass
│   ├── data_loader.py      # HuggingFace dataset loading
│   ├── prompts.py          # Task prompts from paper
│   ├── response_parser.py  # Multi-label response parsing
│   ├── metrics.py          # F1/precision/recall computation
│   ├── vision_client.py    # Unified Ollama/OpenRouter client
│   └── evaluator.py        # Main orchestrator
├── data_generation/
│   ├── pretrain/           # WebDataset tar generators
│   └── sft/                # SFT conversation generators
├── scripts/
│   ├── run_evaluation.py   # CLI entry point
│   ├── download_font.sh    # Download SVG fonts
│   ├── pretrain_data_generation.sh
│   └── sft_data_generation.sh
├── docs/
│   ├── evaluation_plan_v2.md   # Implementation plan
│   ├── model-matches.md        # Paper model mappings
│   └── *.pdf                   # Reference papers
├── results/                # Evaluation output
├── requirements.txt
├── CLAUDE.md              # Claude Code guidance
└── README_mike.md         # This file
```

---

## Model Recommendations

### For Local Evaluation (Ollama)

| Model | Notes |
|-------|-------|
| `llava:7b` | Baseline, widely available |
| `llama3.2-vision:11b` | Good performance |
| `qwen2-vl:7b` | Strong performer |
| `molmo:7b` | Allen AI model |

### For API Evaluation (OpenRouter)

| Model | Performance | Cost per 1K samples |
|-------|-------------|---------------------|
| `openai/gpt-4o` | Best overall | ~$7-11 |
| `qwen/qwen2.5-vl-72b-instruct` | Near GPT-4o | ~$0.36 |
| `anthropic/claude-3.5-sonnet` | Strong | ~$7.50 |
| `meta-llama/llama-3.2-11b-vision-instruct` | Good | ~$0.12 |

**Recommendation**: Use `qwen/qwen2.5-vl-72b-instruct` for best value.

---

## References

- [SalBench Paper (arXiv:2507.04741)](https://arxiv.org/abs/2507.04741)
- [HuggingFace Dataset](https://huggingface.co/datasets/salbench-vlm/salbench)
- [SalBench Website](https://salbench.github.io)

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# For remote servers, ensure the port is accessible
curl http://your-server:11434/api/tags
```

### OpenRouter API Errors

- Verify `OPENROUTER_API_KEY` is set
- Check rate limits at [openrouter.ai](https://openrouter.ai)
- Use `--timeout` for slow models

### Memory Issues with Large Models

72B models require significant GPU memory (>48GB VRAM). Use OpenRouter for these or smaller local models.
