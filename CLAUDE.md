# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SalBench (Saliency Benchmark) is a benchmark for evaluating perceptual capabilities of Vision-Language Models. It tests LVLM ability to detect visually salient features like color, orientation, and size anomalies in images.

## Common Commands

### Setup
```bash
pip install -r requirements.txt
./scripts/download_font.sh  # Downloads SVG fonts required for orientation/size data generation
```

### Data Generation

**Pretraining data** (outputs tar files with images + captions):
```bash
./scripts/pretrain_data_generation.sh
```

**SFT data** (outputs folders with images + conversation JSON):
```bash
./scripts/sft_data_generation.sh
```

**Individual generators** (all support `--main_path`, `--total_images`, `--images_per_tar`/`--images_per_folder`, `--tokenizer_model`):
```bash
python ./data_generation/pretrain/gen_color.py --total_images 1000
python ./data_generation/pretrain/gen_orientation.py --total_images 1000
python ./data_generation/pretrain/gen_size.py --total_images 1000
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

### Key Dependencies

- **svglib/reportlab/pycairo**: SVG rendering for orientation and size generators
- **PIL**: Shape drawing for color generator
- **transformers**: Tokenizer for computing sample token lengths (batching by max_tokens_per_sample)
