"""Configuration for SalBench evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class EvaluationConfig:
    """Configuration for running SalBench evaluation."""

    # Backend settings
    backend: str = "ollama"  # "ollama" or "openrouter"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    model_name: str = "llava:7b"

    # Data settings - use HuggingFace dataset
    dataset_name: str = "salbench-vlm/salbench"
    splits: List[str] = field(default_factory=lambda: ["synthetic", "natural"])

    # Evaluation settings
    task_types: List[str] = field(default_factory=lambda: ["D", "R", "VR"])
    num_shots: int = 0  # 0, 3, or 5
    num_samples: Optional[int] = None  # None = all samples

    # API settings
    timeout_seconds: int = 120
    retry_attempts: int = 3
    temperature: float = 0.0
    max_tokens: int = 256

    # Output settings
    output_dir: str = "./results"

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Set default base_url based on backend
        if self.backend == "openrouter" and self.base_url == "http://localhost:11434":
            self.base_url = "https://openrouter.ai/api/v1"

        # Get API key from environment if not provided
        if self.api_key is None and self.backend == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

        # Validate backend
        if self.backend not in ("ollama", "openrouter"):
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'ollama' or 'openrouter'")

        # Validate num_shots
        if self.num_shots not in (0, 3, 5):
            raise ValueError(f"Invalid num_shots: {self.num_shots}. Must be 0, 3, or 5")

        # Validate splits
        valid_splits = {"synthetic", "natural"}
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(f"Invalid split: {split}. Must be 'synthetic' or 'natural'")

        # Validate task types
        valid_tasks = {"D", "R", "VR"}
        for task in self.task_types:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task type: {task}. Must be 'D', 'R', or 'VR'")

    @property
    def api_endpoint(self) -> str:
        """Get the chat completions API endpoint."""
        if self.backend == "ollama":
            return f"{self.base_url}/v1/chat/completions"
        else:  # openrouter
            return f"{self.base_url}/chat/completions"
