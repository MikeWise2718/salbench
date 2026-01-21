"""Load SalBench benchmark data from HuggingFace."""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from PIL import Image

from datasets import load_dataset

logger = logging.getLogger(__name__)


# Dataset config mapping
# P3 = Synthetic (3 classes: orientation, color, size)
# O3 = Natural (7 classes: orientation, color, focus, shape, size, location, pattern)
CONFIG_MAP = {
    # (split, task) -> config_name
    ("synthetic", "D"): "P3",
    ("synthetic", "R"): "P3_box",
    ("synthetic", "VR"): "P3_box_img",
    ("natural", "D"): "O3",
    ("natural", "R"): "O3_box",
    ("natural", "VR"): "O3_box_img",
}

# Valid labels for each split
SYNTHETIC_LABELS = {"orientation", "color", "size"}
NATURAL_LABELS = {"orientation", "color", "focus", "shape", "size", "location", "pattern"}


@dataclass
class BenchmarkSample:
    """A single benchmark sample from the SalBench dataset."""

    sample_id: str
    split: str  # "synthetic" or "natural"
    task: str  # "D", "R", or "VR"
    image: Image.Image
    question: str  # The prompt from the dataset
    answer: str  # Ground truth answer
    labels: Set[str]  # Parsed ground truth labels
    metadata: Dict[str, Any] = field(default_factory=dict)


class SalBenchDataLoader:
    """Load and prepare SalBench benchmark data from HuggingFace."""

    def __init__(self, dataset_name: str = "salbench-vlm/salbench"):
        self.dataset_name = dataset_name
        self._cache: Dict[str, List[BenchmarkSample]] = {}

    def _parse_answer(self, answer: str, split: str) -> Set[str]:
        """Parse answer string into a set of labels."""
        valid_labels = SYNTHETIC_LABELS if split == "synthetic" else NATURAL_LABELS

        # Normalize and split on common separators
        answer_lower = answer.lower().strip()
        for sep in [",", ";", " and ", "\n"]:
            answer_lower = answer_lower.replace(sep, ",")

        labels = set()
        for part in answer_lower.split(","):
            part = part.strip()
            for valid in valid_labels:
                if valid in part:
                    labels.add(valid)
                    break

        # If no labels found, try exact match
        if not labels:
            answer_clean = answer.lower().strip()
            if answer_clean in valid_labels:
                labels.add(answer_clean)

        return labels

    def load_samples(
        self,
        split: str,
        task: str,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[BenchmarkSample]:
        """
        Load samples for a specific split and task.

        Args:
            split: "synthetic" or "natural"
            task: "D", "R", or "VR"
            num_samples: Optional limit on number of samples
            shuffle: Whether to shuffle samples

        Returns:
            List of BenchmarkSample objects
        """
        cache_key = f"{split}_{task}"

        if cache_key not in self._cache:
            config_name = CONFIG_MAP.get((split, task))
            if not config_name:
                logger.error(f"Invalid split/task combination: {split}/{task}")
                return []

            logger.info(f"Loading {config_name} from HuggingFace...")

            try:
                dataset = load_dataset(self.dataset_name, config_name, split="test")
            except Exception as e:
                logger.error(f"Failed to load dataset config '{config_name}': {e}")
                return []

            samples = []
            for idx, item in enumerate(dataset):
                try:
                    image = item.get("image")
                    if image is None:
                        continue

                    answer = item.get("answer", "")
                    labels = self._parse_answer(answer, split)

                    if not labels:
                        logger.warning(f"Sample {idx} has no valid labels from answer: {answer}")
                        # Still include it with empty labels for completeness

                    sample = BenchmarkSample(
                        sample_id=f"{split}_{task}_{idx}",
                        split=split,
                        task=task,
                        image=image,
                        question=item.get("question", ""),
                        answer=answer,
                        labels=labels,
                        metadata={
                            "image_id": item.get("image_id", ""),
                            "index": idx,
                        },
                    )
                    samples.append(sample)

                except Exception as e:
                    logger.warning(f"Error parsing sample {idx}: {e}")

            logger.info(f"Loaded {len(samples)} samples for {split}/{task}")
            self._cache[cache_key] = samples

        samples = self._cache[cache_key].copy()

        if shuffle:
            random.shuffle(samples)

        if num_samples is not None:
            samples = samples[:num_samples]

        return samples

    def get_samples_for_split(
        self,
        split: str,
        tasks: List[str],
        num_samples: Optional[int] = None,
    ) -> Dict[str, List[BenchmarkSample]]:
        """
        Load samples for all specified tasks in a split.

        Returns:
            Dict mapping task -> list of samples
        """
        result = {}
        for task in tasks:
            result[task] = self.load_samples(split, task, num_samples)
        return result

    @staticmethod
    def get_valid_labels(split: str) -> Set[str]:
        """Get the set of valid labels for a split."""
        return SYNTHETIC_LABELS.copy() if split == "synthetic" else NATURAL_LABELS.copy()

    @staticmethod
    def get_config_name(split: str, task: str) -> Optional[str]:
        """Get the HuggingFace config name for a split/task combination."""
        return CONFIG_MAP.get((split, task))
