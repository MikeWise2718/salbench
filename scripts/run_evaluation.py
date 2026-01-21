#!/usr/bin/env python3
"""CLI entry point for SalBench evaluation."""

import asyncio
import logging
import os
import sys

import rich_click as click

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config import EvaluationConfig
from evaluation.evaluator import SalBenchEvaluator


def setup_logging(verbose: bool):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["ollama", "openrouter"]),
    default="ollama",
    help="API backend to use",
)
@click.option(
    "--base-url",
    default=None,
    help="API base URL (default: auto-detected based on backend)",
)
@click.option(
    "--api-key",
    default=None,
    envvar="OPENROUTER_API_KEY",
    help="API key (required for OpenRouter, uses OPENROUTER_API_KEY env var)",
)
@click.option(
    "--model",
    default="llava:7b",
    help="Model name (e.g., llava:7b for Ollama, qwen/qwen2-vl-72b-instruct for OpenRouter)",
)
@click.option(
    "--dataset",
    default="salbench-vlm/salbench",
    help="HuggingFace dataset name",
)
@click.option(
    "--splits",
    default="synthetic,natural",
    help="Comma-separated splits to evaluate (synthetic, natural)",
)
@click.option(
    "--tasks",
    default="D,R,VR",
    help="Comma-separated task types (D=Detection, R=Referring, VR=Visual Referring)",
)
@click.option(
    "--shots",
    type=click.Choice(["0", "3", "5"]),
    default="0",
    help="Number of few-shot examples",
)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Number of samples per split (default: all)",
)
@click.option(
    "--output",
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help="Request timeout in seconds",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    dataset: str,
    splits: str,
    tasks: str,
    shots: str,
    samples: int,
    output: str,
    timeout: int,
    verbose: bool,
):
    """
    Run SalBench evaluation against vision-language models.

    \b
    This evaluates VLMs on the SalBench benchmark from HuggingFace, testing their
    ability to detect visual saliency features (color, orientation, size, etc.)
    in images.

    \b
    Tasks:
      D  - Detection: Identify which feature type(s) differ in the image
      R  - Referring: Given bbox coordinates in text, classify the feature type
      VR - Visual Referring: Given red-box overlay image, classify the feature type

    \b
    Splits:
      synthetic - P3 dataset with 3 feature classes (Orientation, Color, Size)
      natural   - O3 dataset with 7 feature classes (+Focus, Shape, Location, Pattern)

    \b
    Examples:
      # Local Ollama evaluation on synthetic split
      python run_evaluation.py --backend ollama --base-url http://192.168.1.100:11434 \\
          --model llava:7b --splits synthetic

      # OpenRouter evaluation on both splits
      python run_evaluation.py --backend openrouter --model qwen/qwen2-vl-72b-instruct \\
          --splits synthetic,natural

      # Quick test with 10 samples
      python run_evaluation.py --samples 10 --tasks D --splits synthetic
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Parse list options
    split_list = [s.strip() for s in splits.split(",")]
    task_types = [t.strip() for t in tasks.split(",")]
    num_shots = int(shots)

    # Validate
    if backend == "openrouter" and not api_key:
        raise click.ClickException(
            "API key required for OpenRouter. Set --api-key or OPENROUTER_API_KEY env var."
        )

    # Validate splits
    valid_splits = {"synthetic", "natural"}
    for split in split_list:
        if split not in valid_splits:
            raise click.ClickException(f"Invalid split: {split}. Must be 'synthetic' or 'natural'")

    # Validate tasks
    valid_tasks = {"D", "R", "VR"}
    for task in task_types:
        if task not in valid_tasks:
            raise click.ClickException(f"Invalid task: {task}. Must be 'D', 'R', or 'VR'")

    # Set default base URL based on backend
    if base_url is None:
        if backend == "ollama":
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        else:
            base_url = "https://openrouter.ai/api/v1"

    # Create config
    config = EvaluationConfig(
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        dataset_name=dataset,
        splits=split_list,
        task_types=task_types,
        num_shots=num_shots,
        num_samples=samples,
        output_dir=output,
        timeout_seconds=timeout,
    )

    logger.info("Configuration:")
    logger.info(f"  Backend: {config.backend}")
    logger.info(f"  Base URL: {config.base_url}")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Dataset: {config.dataset_name}")
    logger.info(f"  Splits: {config.splits}")
    logger.info(f"  Tasks: {config.task_types}")
    logger.info(f"  Shots: {config.num_shots}")
    logger.info(f"  Samples: {config.num_samples or 'all'}")

    # Run evaluation
    evaluator = SalBenchEvaluator(config)

    try:
        report = asyncio.run(evaluator.run_evaluation())
        logger.info("Evaluation completed successfully")

        # Print leaderboard row
        row = report["leaderboard_row"]
        print("\nLeaderboard Row:")
        print("-" * 50)
        for key, value in row.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
