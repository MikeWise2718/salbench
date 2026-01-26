# SalBench Evaluation Plan v2 (Revised Based on Paper)

> **STATUS: COMPLETED** - This plan has been fully implemented. See `README_mike.md` for usage instructions.

## Package Manager

This project uses **uv** as the package manager. All commands should be run via `uv run` or `uv pip`.

## Critical Corrections from Paper

The original implementation had several fundamental errors. The arXiv paper (2507.04741) clarifies:

### 1. Data Source
- **Wrong**: Using the repo's generated SFT training data
- **Correct**: Use the **P3/O3 benchmark datasets** available at:
  - HuggingFace: `https://huggingface.co/datasets/salbench-vlm/salbench`
  - Website: `https://salbench.github.io`

### 2. Task Definitions

| Task | Wrong (Original) | Correct (Paper) |
|------|-----------------|-----------------|
| **Detection** | "Is there an anomaly?" → Yes/No | "What feature type differs?" → Classify (color/orientation/size) |
| **Referring** | "Where is it?" → Coordinates | Given bbox in text → Classify feature type |
| **Visual Referring** | Multi-turn conversation | Given red-box image → Classify feature type |

### 3. Output Format
- **Wrong**: Single label or coordinates
- **Correct**: Multi-label classification (e.g., "Color, Size, Pattern")

### 4. Class Sets
- **Synthetic (P3)**: Orientation, Color, Size (3 classes)
- **Natural (O3)**: Orientation, Color, Focus, Shape, Size, Location, Pattern (7 classes)

---

## Revised Implementation Plan

### Step 1: Update Data Loader

**File**: `evaluation/data_loader.py`

```python
from datasets import load_dataset

class SalBenchDataLoader:
    def __init__(self, split="synthetic"):
        # Load from HuggingFace
        self.dataset = load_dataset("salbench-vlm/salbench", split=split)

    def get_detection_samples(self):
        """Get samples for Detection task (original images)."""
        pass

    def get_referring_samples(self):
        """Get samples with bbox annotations for Referring task."""
        pass

    def get_visual_referring_samples(self):
        """Get samples with red-box overlay images for Visual Referring task."""
        pass
```

**Key fields to expect from dataset**:
- `image`: PIL Image or path
- `image_with_box`: PIL Image with red box overlay (for Visual Referring)
- `bbox`: Bounding box coordinates (xmin, ymin, xmax, ymax)
- `label`: Ground truth feature(s) - can be multi-label
- `num_distractors`: Number of distractor objects
- `object_category`: Type of objects in image
- `split`: "synthetic" or "natural"
- `difficulty`: "easy", "medium", "hard" (for synthetic)

### Step 2: Update Prompts

**File**: `evaluation/prompts.py`

```python
# Exact prompts from paper Section 4

SYNTHETIC_FEATURES = "Orientation, Color, Size"
NATURAL_FEATURES = "Orientation, Color, Focus, Shape, Size, Location, Pattern"

DETECTION_PROMPT_TEMPLATE = """Context: Given this list of low-level visual features defined according to feature integration theory: {features}.
Task: Examine the provided image and identify the feature(s) in which one object notably differs from the others.
Write out all applicable features separated by comma."""

REFERRING_PROMPT_TEMPLATE = """Context: This image depicts a scene with {num_distractors} {object_category}. Among those, one {object_category} at location given by this bounding box (xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}), is different from the others.
Given this list of low-level visual features defined according to feature integration theory: {features}.
Task: In which way does the special object differ from the rest. Write out all applicable features separated by comma."""

VISUAL_REFERRING_PROMPT_TEMPLATE = """Context: This image depicts a scene with {num_distractors} {object_category}. Among those, one {object_category} highlighted in a red box is different from the others.
Task: Given this list of low-level visual features defined according to feature integration theory: {features}. In which way does the special object differ from the rest. Write out all applicable feature(s) separated by comma."""


def get_detection_prompt(split: str) -> str:
    features = SYNTHETIC_FEATURES if split == "synthetic" else NATURAL_FEATURES
    return DETECTION_PROMPT_TEMPLATE.format(features=features)


def get_referring_prompt(split: str, num_distractors: int, object_category: str, bbox: dict) -> str:
    features = SYNTHETIC_FEATURES if split == "synthetic" else NATURAL_FEATURES
    return REFERRING_PROMPT_TEMPLATE.format(
        features=features,
        num_distractors=num_distractors,
        object_category=object_category,
        **bbox
    )


def get_visual_referring_prompt(split: str, num_distractors: int, object_category: str) -> str:
    features = SYNTHETIC_FEATURES if split == "synthetic" else NATURAL_FEATURES
    return VISUAL_REFERRING_PROMPT_TEMPLATE.format(
        features=features,
        num_distractors=num_distractors,
        object_category=object_category
    )
```

### Step 3: Update Response Parser

**File**: `evaluation/response_parser.py`

```python
from typing import Set

VALID_SYNTHETIC_LABELS = {"orientation", "color", "size"}
VALID_NATURAL_LABELS = {"orientation", "color", "focus", "shape", "size", "location", "pattern"}


class ResponseParser:
    def parse_multi_label_response(self, response: str, split: str) -> Set[str]:
        """
        Parse comma-separated feature labels from model response.

        Returns:
            Set of normalized label strings
        """
        valid_labels = VALID_SYNTHETIC_LABELS if split == "synthetic" else VALID_NATURAL_LABELS

        # Normalize and split
        response_lower = response.lower().strip()

        # Handle various separators
        for sep in [",", ";", "and", "\n"]:
            response_lower = response_lower.replace(sep, ",")

        # Extract labels
        predicted = set()
        for part in response_lower.split(","):
            part = part.strip()
            # Match against valid labels
            for valid in valid_labels:
                if valid in part:
                    predicted.add(valid)
                    break

        return predicted
```

### Step 4: Update Metrics

**File**: `evaluation/metrics.py`

```python
from typing import Set, List, Dict
from dataclasses import dataclass


@dataclass
class MultiLabelResult:
    sample_id: str
    split: str
    task: str
    predicted: Set[str]
    ground_truth: Set[str]
    exact_match: bool
    precision: float
    recall: float
    f1: float


class MetricsCalculator:
    def compute_multi_label_metrics(
        self,
        predicted: Set[str],
        ground_truth: Set[str]
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for multi-label prediction."""
        if not predicted and not ground_truth:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}

        if not predicted or not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

        intersection = predicted & ground_truth
        precision = len(intersection) / len(predicted)
        recall = len(intersection) / len(ground_truth)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        exact_match = 1.0 if predicted == ground_truth else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match
        }

    def aggregate_results(self, results: List[MultiLabelResult]) -> Dict:
        """Aggregate results by split, task, and category."""
        # Group by split and task
        aggregated = {}

        for result in results:
            key = (result.split, result.task)
            if key not in aggregated:
                aggregated[key] = {
                    "total": 0,
                    "exact_match_sum": 0,
                    "f1_sum": 0,
                    "per_category": {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in
                                    (VALID_SYNTHETIC_LABELS if result.split == "synthetic" else VALID_NATURAL_LABELS)}
                }

            agg = aggregated[key]
            agg["total"] += 1
            agg["exact_match_sum"] += result.exact_match
            agg["f1_sum"] += result.f1

            # Per-category stats for macro F1
            for cat in result.ground_truth:
                if cat in result.predicted:
                    agg["per_category"][cat]["tp"] += 1
                else:
                    agg["per_category"][cat]["fn"] += 1
            for cat in result.predicted:
                if cat not in result.ground_truth:
                    agg["per_category"][cat]["fp"] += 1

        return aggregated
```

### Step 5: Update Evaluator

**File**: `evaluation/evaluator.py`

```python
class SalBenchEvaluator:
    async def run_detection(self, client, sample, split):
        """Detection task: classify which feature type differs."""
        prompt = get_detection_prompt(split)

        response = await client.send_request(
            sample["image"],  # Original image, no box
            prompt
        )

        predicted = self.parser.parse_multi_label_response(response.content, split)
        ground_truth = set(sample["label"])  # Can be multiple labels

        return self.metrics.compute_multi_label_metrics(predicted, ground_truth)

    async def run_referring(self, client, sample, split):
        """Referring task: given bbox coords in text, classify feature."""
        prompt = get_referring_prompt(
            split=split,
            num_distractors=sample["num_distractors"],
            object_category=sample["object_category"],
            bbox=sample["bbox"]
        )

        response = await client.send_request(
            sample["image"],  # Original image, no box
            prompt
        )

        predicted = self.parser.parse_multi_label_response(response.content, split)
        ground_truth = set(sample["label"])

        return self.metrics.compute_multi_label_metrics(predicted, ground_truth)

    async def run_visual_referring(self, client, sample, split):
        """Visual Referring task: given red-box image, classify feature."""
        prompt = get_visual_referring_prompt(
            split=split,
            num_distractors=sample["num_distractors"],
            object_category=sample["object_category"]
        )

        response = await client.send_request(
            sample["image_with_box"],  # Image WITH red box overlay
            prompt
        )

        predicted = self.parser.parse_multi_label_response(response.content, split)
        ground_truth = set(sample["label"])

        return self.metrics.compute_multi_label_metrics(predicted, ground_truth)
```

### Step 6: Update Config

**File**: `evaluation/config.py`

```python
@dataclass
class EvaluationConfig:
    # Backend
    backend: str = "ollama"  # or "openrouter"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    model_name: str = "llava:7b"

    # Data - NEW: use HuggingFace dataset
    dataset_name: str = "salbench-vlm/salbench"
    splits: List[str] = field(default_factory=lambda: ["synthetic", "natural"])

    # Evaluation
    tasks: List[str] = field(default_factory=lambda: ["D", "R", "VR"])
    num_shots: int = 0
    num_samples: Optional[int] = None  # None = all samples

    # Output
    output_dir: str = "./results"
```

### Step 7: Update CLI

**File**: `scripts/run_evaluation.py`

```bash
# Example usage (all commands use uv):

# Evaluate on synthetic split only
uv run python scripts/run_evaluation.py \
    --backend ollama \
    --base-url http://YOUR_HOST:11434 \
    --model llava:7b \
    --splits synthetic \
    --tasks D,R,VR

# Evaluate on both splits with OpenRouter
uv run python scripts/run_evaluation.py \
    --backend openrouter \
    --model qwen/qwen2-vl-72b-instruct \
    --splits synthetic,natural \
    --tasks D,R,VR \
    --num-samples 100

# Quick test
uv run python scripts/run_evaluation.py --samples 5 --tasks D --splits synthetic -v
```

### Step 8: Update Requirements

**File**: `requirements.txt`

```bash
# Install dependencies with uv:
uv pip install -r requirements.txt

# Or install just the new dependency:
uv pip install datasets>=2.14.0
```

---

## Output Format (Matching Paper's Table 1)

```csv
Model,Shot,Detection_NAT,Detection_SYN,Referring_NAT,Referring_SYN,VisualRef_NAT,VisualRef_SYN
llava:7b,0,24.5,16.3,21.4,10.1,20.8,16.6
qwen2-vl:7b,0,32.5,55.7,32.5,34.2,35.2,57.4
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `evaluation/config.py` | Add `dataset_name`, `splits` options |
| `evaluation/data_loader.py` | Complete rewrite for HuggingFace dataset |
| `evaluation/prompts.py` | Complete rewrite with exact paper prompts |
| `evaluation/response_parser.py` | Add `parse_multi_label_response()` |
| `evaluation/metrics.py` | Add multi-label F1 computation |
| `evaluation/evaluator.py` | Update all three task methods |
| `scripts/run_evaluation.py` | Update CLI options |
| `requirements.txt` | Add `datasets>=2.14.0` |

---

## Validation Checklist

1. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Data loading**: Verify HuggingFace dataset loads correctly:
   ```bash
   uv run python -c "from datasets import load_dataset; ds = load_dataset('salbench-vlm/salbench', split='synthetic'); print(len(ds), ds[0].keys())"
   ```

3. **Quick validation**: Run a quick test:
   ```bash
   uv run python scripts/run_evaluation.py --samples 5 --tasks D --splits synthetic -v
   ```

4. **Prompts**: Compare generated prompts with paper's exact wording

5. **Response parsing**: Test with various model outputs (comma-separated, verbose, etc.)

6. **Metrics**: Validate F1 calculation matches paper's methodology

7. **Baseline**: Compare LLaVA 1.6-7B results with paper's Table 1:
   ```bash
   uv run python scripts/run_evaluation.py --backend ollama --model llava:7b --splits synthetic
   ```
   Expected results:
   - Detection SYN: ~16.3% F1
   - Referring SYN: ~10.1% F1
   - Visual Referring SYN: ~16.6% F1

---

## Actual Results (January 2026)

We ran the evaluation framework on several models. Results on the **synthetic (P3)** split:

| Model | Detection | Referring | VisualRef | Cost | Notes |
|-------|-----------|-----------|-----------|------|-------|
| openai/gpt-4o | 86.3% | 86.7% | 76.3% | $16.25 | Full run (2589 samples/task) |
| qwen/qwen2.5-vl-72b-instruct | 67.0% | 88.7% | 78.6% | $1.70 | Full run (2589 samples/task) |
| anthropic/claude-3.5-sonnet | 65.0% | 70.0% | 50.0% | $0.14 | Small sample (10 samples) |

**Key Observations:**
- GPT-4o performs best on Detection, but Qwen2-VL-72B is competitive on Referring/VisualRef
- Qwen2-VL-72B offers ~10x better cost efficiency than GPT-4o
- All models show lower performance on Visual Referring vs Referring

---

## Key Differences Summary

| Aspect | Original Implementation | Revised (Paper-Based) |
|--------|------------------------|----------------------|
| Data source | Generated SFT data | P3/O3 via HuggingFace |
| Detection output | Yes/No | Feature class(es) |
| Referring input | Coordinates in answer | Coordinates in prompt |
| Visual Referring | Multi-turn conversation | Single turn with red-box image |
| Labels | Single-label | Multi-label |
| Metrics | Position-based F1 | Category-based F1 |
