"""Main evaluation orchestrator for SalBench."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from tqdm import tqdm

from .config import EvaluationConfig
from .data_loader import SalBenchDataLoader, BenchmarkSample
from .vision_client import VisionClient
from .response_parser import ResponseParser
from .metrics import (
    MetricsCalculator,
    MultiLabelResult,
    format_results_table,
    format_detailed_report,
)

logger = logging.getLogger(__name__)
console = Console()


# OpenRouter pricing per 1M tokens (approximate, varies by model)
OPENROUTER_PRICING = {
    # Format: "model_pattern": (input_cost_per_1m, output_cost_per_1m)
    "qwen/qwen2-vl-72b": (0.40, 0.40),
    "qwen/qwen2-vl-7b": (0.10, 0.10),
    "google/gemini-flash": (0.075, 0.30),
    "google/gemini-pro": (1.25, 5.00),
    "anthropic/claude-3": (3.00, 15.00),
    "anthropic/claude-3.5": (3.00, 15.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "meta-llama/llama-3": (0.10, 0.10),
    "default": (1.00, 3.00),  # Conservative default
}


@dataclass
class UsageStats:
    """Track token usage and timing statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def estimate_cost(self, model_name: str) -> Optional[float]:
        """Estimate cost for OpenRouter based on model pricing."""
        # Find matching pricing
        pricing = OPENROUTER_PRICING.get("default")
        for pattern, costs in OPENROUTER_PRICING.items():
            if pattern != "default" and pattern in model_name.lower():
                pricing = costs
                break

        input_cost, output_cost = pricing
        cost = (self.input_tokens / 1_000_000 * input_cost +
                self.output_tokens / 1_000_000 * output_cost)
        return cost


class SalBenchEvaluator:
    """Main evaluator for running SalBench benchmark."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.data_loader = SalBenchDataLoader(config.dataset_name)
        self.parser = ResponseParser()
        self.metrics = MetricsCalculator()
        self.results: List[MultiLabelResult] = []
        self.usage_stats = UsageStats()

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run the full evaluation pipeline."""
        logger.info(f"Starting evaluation with model: {self.config.model_name}")
        logger.info(f"Backend: {self.config.backend}")
        logger.info(f"Splits: {self.config.splits}")
        logger.info(f"Tasks: {self.config.task_types}")
        logger.info(f"Samples per task: {self.config.num_samples or 'all'}")
        logger.info(f"Few-shot: {self.config.num_shots}")

        self.results = []
        self.usage_stats = UsageStats()  # Reset stats for this run

        async with VisionClient(self.config) as client:
            for split in self.config.splits:
                logger.info(f"Evaluating {split} split...")

                for task in self.config.task_types:
                    samples = self.data_loader.load_samples(
                        split=split,
                        task=task,
                        num_samples=self.config.num_samples,
                    )

                    if not samples:
                        logger.warning(f"No samples found for {split}/{task}")
                        continue

                    logger.info(f"Evaluating {len(samples)} samples for {split}/{task}")

                    task_results = await self._evaluate_samples(
                        client, samples, split, task
                    )
                    self.results.extend(task_results)

        # Mark end time
        self.usage_stats.end_time = time.time()

        # Compile and save results
        report = self._compile_report()
        self._save_results(report)

        return report

    async def _evaluate_samples(
        self,
        client: VisionClient,
        samples: List[BenchmarkSample],
        split: str,
        task: str,
    ) -> List[MultiLabelResult]:
        """Evaluate all samples for a specific split/task."""
        results = []
        desc = f"{split}/{task}"

        for sample in tqdm(samples, desc=desc):
            try:
                result = await self._evaluate_single(client, sample, split)
                results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating sample {sample.sample_id}: {e}")
                results.append(self.metrics.create_result(
                    sample_id=sample.sample_id,
                    split=split,
                    task=task,
                    predicted=set(),
                    ground_truth=sample.labels,
                    raw_response="",
                    error=str(e),
                ))

        return results

    async def _evaluate_single(
        self,
        client: VisionClient,
        sample: BenchmarkSample,
        split: str,
    ) -> MultiLabelResult:
        """
        Evaluate a single sample.

        Uses the question/prompt directly from the dataset.
        """
        # Send request using the question from the dataset
        response = await client.send_request(
            sample.image,
            sample.question,
            few_shot_examples=None,  # TODO: implement few-shot if needed
        )

        # Track token usage
        self.usage_stats.total_requests += 1
        if response.usage:
            self.usage_stats.input_tokens += response.usage.get("prompt_tokens", 0)
            self.usage_stats.output_tokens += response.usage.get("completion_tokens", 0)

        if not response.success:
            self.usage_stats.failed_requests += 1
            return self.metrics.create_result(
                sample_id=sample.sample_id,
                split=split,
                task=sample.task,
                predicted=set(),
                ground_truth=sample.labels,
                raw_response=response.content,
                error=response.error,
            )

        # Parse the model's response
        predicted = self.parser.parse_multi_label_response(response.content, split)

        return self.metrics.create_result(
            sample_id=sample.sample_id,
            split=split,
            task=sample.task,
            predicted=predicted,
            ground_truth=sample.labels,
            raw_response=response.content,
        )

    def _compile_report(self) -> Dict[str, Any]:
        """Compile evaluation results into a report."""
        aggregated = self.metrics.aggregate_results(self.results)

        leaderboard_row = format_results_table(
            aggregated,
            self.config.model_name,
            self.config.num_shots,
        )

        detailed_report = format_detailed_report(aggregated)

        # Build usage stats dict
        usage_dict = {
            "elapsed_seconds": round(self.usage_stats.elapsed_seconds, 1),
            "total_requests": self.usage_stats.total_requests,
            "failed_requests": self.usage_stats.failed_requests,
            "input_tokens": self.usage_stats.input_tokens,
            "output_tokens": self.usage_stats.output_tokens,
            "total_tokens": self.usage_stats.total_tokens,
        }

        # Add cost estimate for OpenRouter
        if self.config.backend == "openrouter" and self.usage_stats.total_tokens > 0:
            usage_dict["estimated_cost_usd"] = round(
                self.usage_stats.estimate_cost(self.config.model_name), 4
            )

        report = {
            "config": {
                "model": self.config.model_name,
                "backend": self.config.backend,
                "dataset": self.config.dataset_name,
                "splits": self.config.splits,
                "tasks": self.config.task_types,
                "num_shots": self.config.num_shots,
                "num_samples": self.config.num_samples,
            },
            "usage": usage_dict,
            "leaderboard_row": leaderboard_row,
            "detailed_metrics": detailed_report,
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _save_results(self, report: Dict[str, Any]):
        """Save evaluation results to disk."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = self.config.model_name.replace("/", "_").replace(":", "_")
        output_dir = Path(self.config.output_dir) / f"{model_safe}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # Save raw responses
        raw_responses_path = output_dir / "raw_responses.json"
        raw_data = [
            {
                "sample_id": r.sample_id,
                "split": r.split,
                "task": r.task,
                "predicted": sorted(r.predicted),
                "ground_truth": sorted(r.ground_truth),
                "exact_match": r.exact_match,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "raw_response": r.raw_response,
                "error": r.error,
            }
            for r in self.results
        ]
        with open(raw_responses_path, "w") as f:
            json.dump(raw_data, f, indent=2)
        logger.info(f"Saved raw responses to {raw_responses_path}")

        # Save leaderboard CSV row
        csv_path = output_dir / "leaderboard_row.csv"
        row = report["leaderboard_row"]
        headers = list(row.keys())
        values = [str(row[h]) for h in headers]
        with open(csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            f.write(",".join(values) + "\n")
        logger.info(f"Saved leaderboard row to {csv_path}")

        # Print colorized summary using rich
        self._print_rich_summary(report, output_dir)

    def _print_rich_summary(self, report: Dict[str, Any], output_dir: Path):
        """Print colorized evaluation summary using rich."""
        console.print()

        # Header panel
        header = Text("EVALUATION COMPLETE", style="bold green")
        console.print(Panel(header, border_style="green"))

        # Config info
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Key", style="cyan")
        config_table.add_column("Value", style="white")
        config_table.add_row("Model", self.config.model_name)
        config_table.add_row("Backend", self.config.backend)
        config_table.add_row("Shots", str(self.config.num_shots))
        console.print(config_table)
        console.print()

        # Usage stats
        usage = report.get("usage", {})
        usage_table = Table(title="[bold cyan]Usage Statistics", box=None, padding=(0, 2))
        usage_table.add_column("Metric", style="cyan")
        usage_table.add_column("Value", style="yellow", justify="right")

        elapsed = usage.get("elapsed_seconds", 0)
        usage_table.add_row("Elapsed Time", f"{elapsed:.1f}s")
        usage_table.add_row("Total Requests", str(usage.get("total_requests", 0)))
        if usage.get("failed_requests", 0) > 0:
            usage_table.add_row("Failed Requests", f"[red]{usage.get('failed_requests', 0)}[/red]")

        # Token usage
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens > 0:
            usage_table.add_row("Input Tokens", f"{input_tokens:,}")
            usage_table.add_row("Output Tokens", f"{output_tokens:,}")
            usage_table.add_row("Total Tokens", f"[bold]{total_tokens:,}[/bold]")

        # Cost estimate for OpenRouter
        if "estimated_cost_usd" in usage:
            cost = usage["estimated_cost_usd"]
            usage_table.add_row("Estimated Cost", f"[green]${cost:.4f}[/green]")

        console.print(usage_table)
        console.print()

        # Results table
        results_table = Table(title="[bold cyan]Results (F1 %)", border_style="blue")
        results_table.add_column("Split", style="cyan", justify="left")
        results_table.add_column("Detection", style="white", justify="right")
        results_table.add_column("Referring", style="white", justify="right")
        results_table.add_column("VisualRef", style="white", justify="right")

        for split in self.config.splits:
            det = report["detailed_metrics"].get(split, {}).get("D", {}).get("f1", 0)
            ref = report["detailed_metrics"].get(split, {}).get("R", {}).get("f1", 0)
            vr = report["detailed_metrics"].get(split, {}).get("VR", {}).get("f1", 0)

            # Color-code based on score
            def score_style(score: float) -> str:
                if score >= 70:
                    return f"[green]{score:.1f}[/green]"
                elif score >= 40:
                    return f"[yellow]{score:.1f}[/yellow]"
                else:
                    return f"[red]{score:.1f}[/red]"

            results_table.add_row(
                split,
                score_style(det),
                score_style(ref),
                score_style(vr),
            )

        console.print(results_table)
        console.print()

        # Output path
        console.print(f"[dim]Results saved to:[/dim] [bold blue]{output_dir}[/bold blue]")
        console.print()
