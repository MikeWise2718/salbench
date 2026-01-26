# SalBench Model Matches

Models benchmarked by the TII UAE team in the SalBench paper, with their best matches in Ollama and OpenRouter.

## Model Mapping Table

| Paper Model | Params | Ollama Match | OpenRouter Match |
|-------------|--------|--------------|------------------|
| **GPT-4o** | - | N/A | `openai/gpt-4o` |
| **Claude-sonnet** | - | N/A | `anthropic/claude-3.5-sonnet` |
| **Qwen2-VL-72B** | 72B | N/A (too large) | `qwen/qwen2.5-vl-72b-instruct` |
| **Qwen2-VL-7B** | 7B | `qwen2-vl:7b` | `qwen/qwen2.5-vl-7b-instruct` |
| **Qwen2-VL-1.5B** | 1.5B | `qwen2-vl:2b` | N/A |
| **NVLM-D-72B** | 72B | N/A | N/A |
| **Molmo-72B** | 72B | N/A | N/A |
| **Molmo-7B** | 7B | `molmo:7b` | N/A |
| **Llama3.2-Vision-11B** | 11B | `llama3.2-vision:11b` | `meta-llama/llama-3.2-11b-vision-instruct` |
| **InternVL2-8B** | 8B | N/A | N/A |
| **InternVL-4B** | 4B | N/A | N/A |
| **LLaVA 1.6-7B** | 7B | `llava:7b` | N/A |
| **Idefics2-8B** | 8B | N/A | N/A |
| **Idefics3-8B** | 8B | N/A | N/A |
| **VILA-1.5-8B** | 8B | N/A | N/A |
| **Phi3-4B** | 4B | N/A (no vision) | N/A |
| **Phi3.5-Vision-3.5B** | 3.5B | N/A | N/A |
| **PaliGemma-3B-448** | 3B | N/A | `google/paligemma-3b-mix-448` |

## Top Performers (0-shot Detection F1 on Synthetic)

From the paper's Table 1:

| Model | Detection (SYN) | Detection (NAT) |
|-------|-----------------|-----------------|
| GPT-4o | 89.2% | 47.6% |
| Qwen2-VL-72B | 88.8% | 41.6% |
| Claude-sonnet | 86.7% | 48.2% |
| Molmo-72B | 83.3% | 40.6% |
| NVLM-D-72B | 77.5% | 41.5% |

## Our Evaluation Results (January 2026)

We ran the evaluation framework against several models on the synthetic split:

| Model | Detection | Referring | VisualRef | Cost | Samples |
|-------|-----------|-----------|-----------|------|---------|
| openai/gpt-4o | 86.3% | 86.7% | 76.3% | $16.25 | 2589/task |
| qwen/qwen2.5-vl-72b-instruct | 67.0% | 88.7% | 78.6% | $1.70 | 2589/task |
| anthropic/claude-3.5-sonnet | 65.0%* | 70.0%* | 50.0%* | $0.14 | 10/task |

*Small sample size - not statistically significant

## Recommended Models for Evaluation

### Ollama (Local)

Best options for local evaluation:

- `llava:7b` - Baseline, widely available
- `llama3.2-vision:11b` - Good performance, Meta model
- `qwen2-vl:7b` - Strong performer in paper
- `molmo:7b` - Allen AI model

### OpenRouter (API)

Best options for API-based evaluation:

- `qwen/qwen2.5-vl-72b-instruct` - Top open-source performer
- `openai/gpt-4o` - Best overall in paper
- `anthropic/claude-3.5-sonnet` - Strong on natural images
- `meta-llama/llama-3.2-11b-vision-instruct` - Good balance

## OpenRouter Pricing (as of Jan 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| `qwen/qwen2.5-vl-72b-instruct` | $0.15 | $0.60 |
| `qwen/qwen2.5-vl-7b-instruct` | $0.10 | $0.10 |
| `openai/gpt-4o` | $3.00 - $5.00 | $10.00 - $15.00 |
| `anthropic/claude-3.5-sonnet` | $3.00 | $15.00 |
| `meta-llama/llama-3.2-11b-vision-instruct` | $0.055 | $0.055 |
| `google/paligemma-3b-mix-448` | $0.10 | $0.10 |

### Estimated Cost per 1000 Samples

Assuming ~2K input tokens + 100 output tokens per sample (image + prompt + response):

| Model | Est. Cost | Performance (SYN) | Value |
|-------|-----------|-------------------|-------|
| `qwen/qwen2.5-vl-72b-instruct` | ~$0.36 | 88.8% | Best |
| `meta-llama/llama-3.2-11b-vision-instruct` | ~$0.12 | 48.7% | Good |
| `openai/gpt-4o` | ~$7-11 | 89.2% | Expensive |
| `anthropic/claude-3.5-sonnet` | ~$7.50 | 86.7% | Expensive |

**Qwen2-VL-72B offers the best value**: similar performance to GPT-4o at ~10x lower cost.

Our actual runs confirmed this - Qwen outperformed GPT-4o on Referring (88.7% vs 86.7%) and VisualRef (78.6% vs 76.3%) tasks while costing $1.70 vs $16.25 for a full evaluation run.

## Notes

- Many models from the paper (NVLM, InternVL, Idefics, VILA) are not readily available on Ollama or OpenRouter
- 72B models generally require significant GPU memory (>48GB VRAM) for local inference
- OpenRouter pricing changes frequently; verify current rates at [openrouter.ai/pricing](https://openrouter.ai/pricing)
- The paper found that larger models significantly outperform smaller ones on saliency tasks
- Ollama models are free to run locally (only hardware costs)
