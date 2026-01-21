"""Task-specific prompts for SalBench evaluation.

Prompts are based on the exact wording from the SalBench paper (arXiv 2507.04741).
"""

from typing import Dict

# Feature lists for each split
SYNTHETIC_FEATURES = "Orientation, Color, Size"
NATURAL_FEATURES = "Orientation, Color, Focus, Shape, Size, Location, Pattern"


# Detection Task Prompt Template
# Task: Classify which feature type(s) differ in the image
DETECTION_PROMPT_TEMPLATE = """Context: Given this list of low-level visual features defined according to feature integration theory: {features}.
Task: Examine the provided image and identify the feature(s) in which one object notably differs from the others.
Write out all applicable features separated by comma."""


# Referring Task Prompt Template
# Task: Given bbox coords in the prompt, classify which feature type(s) differ
REFERRING_PROMPT_TEMPLATE = """Context: This image depicts a scene with {num_distractors} {object_category}. Among those, one {object_category} at location given by this bounding box (xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}), is different from the others.
Given this list of low-level visual features defined according to feature integration theory: {features}.
Task: In which way does the special object differ from the rest. Write out all applicable features separated by comma."""


# Visual Referring Task Prompt Template
# Task: Given red-box overlay image, classify which feature type(s) differ
VISUAL_REFERRING_PROMPT_TEMPLATE = """Context: This image depicts a scene with {num_distractors} {object_category}. Among those, one {object_category} highlighted in a red box is different from the others.
Task: Given this list of low-level visual features defined according to feature integration theory: {features}. In which way does the special object differ from the rest. Write out all applicable feature(s) separated by comma."""


def get_features_list(split: str) -> str:
    """Get the comma-separated features list for a split."""
    return SYNTHETIC_FEATURES if split == "synthetic" else NATURAL_FEATURES


def get_detection_prompt(split: str) -> str:
    """
    Get the Detection task prompt.

    Args:
        split: "synthetic" or "natural"

    Returns:
        Formatted prompt string
    """
    features = get_features_list(split)
    return DETECTION_PROMPT_TEMPLATE.format(features=features)


def get_referring_prompt(
    split: str,
    num_distractors: int,
    object_category: str,
    bbox: Dict[str, float],
) -> str:
    """
    Get the Referring task prompt.

    Args:
        split: "synthetic" or "natural"
        num_distractors: Number of distractor objects
        object_category: Type of objects (e.g., "icon", "shape")
        bbox: Dict with xmin, ymin, xmax, ymax

    Returns:
        Formatted prompt string
    """
    features = get_features_list(split)
    return REFERRING_PROMPT_TEMPLATE.format(
        features=features,
        num_distractors=num_distractors,
        object_category=object_category,
        xmin=bbox.get("xmin", 0),
        ymin=bbox.get("ymin", 0),
        xmax=bbox.get("xmax", 0),
        ymax=bbox.get("ymax", 0),
    )


def get_visual_referring_prompt(
    split: str,
    num_distractors: int,
    object_category: str,
) -> str:
    """
    Get the Visual Referring task prompt.

    Args:
        split: "synthetic" or "natural"
        num_distractors: Number of distractor objects
        object_category: Type of objects

    Returns:
        Formatted prompt string
    """
    features = get_features_list(split)
    return VISUAL_REFERRING_PROMPT_TEMPLATE.format(
        features=features,
        num_distractors=num_distractors,
        object_category=object_category,
    )


# Few-shot example templates
def get_few_shot_detection_example(split: str, labels: str) -> Dict[str, str]:
    """Get a few-shot example for detection task."""
    return {
        "question": get_detection_prompt(split),
        "answer": labels,  # e.g., "Color" or "Orientation, Size"
    }


def get_few_shot_referring_example(
    split: str,
    num_distractors: int,
    object_category: str,
    bbox: Dict[str, float],
    labels: str,
) -> Dict[str, str]:
    """Get a few-shot example for referring task."""
    return {
        "question": get_referring_prompt(split, num_distractors, object_category, bbox),
        "answer": labels,
    }


def get_few_shot_visual_referring_example(
    split: str,
    num_distractors: int,
    object_category: str,
    labels: str,
) -> Dict[str, str]:
    """Get a few-shot example for visual referring task."""
    return {
        "question": get_visual_referring_prompt(split, num_distractors, object_category),
        "answer": labels,
    }
