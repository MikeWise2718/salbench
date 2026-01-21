"""Metrics computation for SalBench evaluation with multi-label support."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any

from .response_parser import VALID_SYNTHETIC_LABELS, VALID_NATURAL_LABELS


@dataclass
class MultiLabelResult:
    """Result of a single multi-label evaluation."""

    sample_id: str
    split: str  # "synthetic" or "natural"
    task: str  # "D", "R", or "VR"
    predicted: Set[str]
    ground_truth: Set[str]
    exact_match: float  # 1.0 if perfect match, 0.0 otherwise
    precision: float
    recall: float
    f1: float
    raw_response: str = ""
    error: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple evaluations."""

    total_samples: int = 0
    successful_samples: int = 0
    exact_match_sum: float = 0.0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    f1_sum: float = 0.0
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def exact_match(self) -> float:
        """Average exact match score as percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.exact_match_sum / self.total_samples) * 100

    @property
    def precision(self) -> float:
        """Average precision as percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.precision_sum / self.total_samples) * 100

    @property
    def recall(self) -> float:
        """Average recall as percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.recall_sum / self.total_samples) * 100

    @property
    def f1_score(self) -> float:
        """Average F1 score as percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.f1_sum / self.total_samples) * 100

    @property
    def success_rate(self) -> float:
        """Percentage of successfully processed samples."""
        if self.total_samples == 0:
            return 0.0
        return (self.successful_samples / self.total_samples) * 100


class MetricsCalculator:
    """Calculate multi-label evaluation metrics for SalBench."""

    def compute_multi_label_metrics(
        self,
        predicted: Set[str],
        ground_truth: Set[str],
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for multi-label prediction.

        Args:
            predicted: Set of predicted labels
            ground_truth: Set of ground truth labels

        Returns:
            Dict with precision, recall, f1, exact_match
        """
        # Handle edge cases
        if not predicted and not ground_truth:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}

        if not predicted or not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

        # Compute intersection
        intersection = predicted & ground_truth

        # Precision = |intersection| / |predicted|
        precision = len(intersection) / len(predicted)

        # Recall = |intersection| / |ground_truth|
        recall = len(intersection) / len(ground_truth)

        # F1 = 2 * precision * recall / (precision + recall)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Exact match: 1.0 if predicted == ground_truth exactly
        exact_match = 1.0 if predicted == ground_truth else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
        }

    def create_result(
        self,
        sample_id: str,
        split: str,
        task: str,
        predicted: Set[str],
        ground_truth: Set[str],
        raw_response: str = "",
        error: Optional[str] = None,
    ) -> MultiLabelResult:
        """Create a MultiLabelResult with computed metrics."""
        if error:
            return MultiLabelResult(
                sample_id=sample_id,
                split=split,
                task=task,
                predicted=predicted,
                ground_truth=ground_truth,
                exact_match=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                raw_response=raw_response,
                error=error,
            )

        metrics = self.compute_multi_label_metrics(predicted, ground_truth)

        return MultiLabelResult(
            sample_id=sample_id,
            split=split,
            task=task,
            predicted=predicted,
            ground_truth=ground_truth,
            exact_match=metrics["exact_match"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            raw_response=raw_response,
        )

    def aggregate_results(
        self,
        results: List[MultiLabelResult],
    ) -> Dict[tuple, AggregatedMetrics]:
        """
        Aggregate results by (split, task) combination.

        Returns:
            Dict mapping (split, task) to AggregatedMetrics
        """
        aggregated: Dict[tuple, AggregatedMetrics] = {}

        for result in results:
            key = (result.split, result.task)

            if key not in aggregated:
                # Initialize per-category tracking
                valid_labels = (
                    VALID_SYNTHETIC_LABELS if result.split == "synthetic"
                    else VALID_NATURAL_LABELS
                )
                per_category = {
                    cat: {"tp": 0, "fp": 0, "fn": 0}
                    for cat in valid_labels
                }
                aggregated[key] = AggregatedMetrics(per_category=per_category)

            agg = aggregated[key]
            agg.total_samples += 1

            if result.error is None:
                agg.successful_samples += 1
                agg.exact_match_sum += result.exact_match
                agg.precision_sum += result.precision
                agg.recall_sum += result.recall
                agg.f1_sum += result.f1

                # Update per-category stats for macro F1
                for cat in result.ground_truth:
                    if cat in agg.per_category:
                        if cat in result.predicted:
                            agg.per_category[cat]["tp"] += 1
                        else:
                            agg.per_category[cat]["fn"] += 1

                for cat in result.predicted:
                    if cat in agg.per_category and cat not in result.ground_truth:
                        agg.per_category[cat]["fp"] += 1

        return aggregated

    def compute_macro_f1(self, aggregated: AggregatedMetrics) -> float:
        """Compute macro F1 from per-category stats."""
        f1_scores = []

        for cat, stats in aggregated.per_category.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            f1_scores.append(f1)

        if not f1_scores:
            return 0.0

        return sum(f1_scores) / len(f1_scores) * 100


def format_results_table(
    aggregated: Dict[tuple, AggregatedMetrics],
    model_name: str,
    num_shots: int,
) -> Dict[str, Any]:
    """
    Format results as a table row matching paper's Table 1 format.

    Returns:
        Dict with columns: Model, Shot, Detection_NAT, Detection_SYN, etc.
    """
    row = {
        "Model": model_name,
        "Shot": num_shots,
    }

    # Map task names
    task_names = {"D": "Detection", "R": "Referring", "VR": "VisualRef"}
    split_names = {"synthetic": "SYN", "natural": "NAT"}

    for (split, task), metrics in aggregated.items():
        task_name = task_names.get(task, task)
        split_name = split_names.get(split, split)
        col_name = f"{task_name}_{split_name}"
        row[col_name] = round(metrics.f1_score, 1)

    return row


def format_detailed_report(
    aggregated: Dict[tuple, AggregatedMetrics],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Format detailed metrics report.

    Returns:
        Nested dict: {split: {task: {metric: value}}}
    """
    report = {}

    for (split, task), metrics in aggregated.items():
        if split not in report:
            report[split] = {}

        report[split][task] = {
            "total_samples": metrics.total_samples,
            "successful_samples": metrics.successful_samples,
            "success_rate": round(metrics.success_rate, 1),
            "exact_match": round(metrics.exact_match, 1),
            "precision": round(metrics.precision, 1),
            "recall": round(metrics.recall, 1),
            "f1": round(metrics.f1_score, 1),
        }

    return report
