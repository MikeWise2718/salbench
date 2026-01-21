"""Parse model responses to extract multi-label classifications."""

import re
from typing import Set

# Valid labels for each split
VALID_SYNTHETIC_LABELS = {"orientation", "color", "size"}
VALID_NATURAL_LABELS = {"orientation", "color", "focus", "shape", "size", "location", "pattern"}


class ResponseParser:
    """Parser for extracting multi-label feature classifications from model responses."""

    def parse_multi_label_response(self, response: str, split: str) -> Set[str]:
        """
        Parse comma-separated feature labels from model response.

        Args:
            response: Raw model response text
            split: "synthetic" or "natural" to determine valid labels

        Returns:
            Set of normalized label strings
        """
        valid_labels = VALID_SYNTHETIC_LABELS if split == "synthetic" else VALID_NATURAL_LABELS

        # Normalize response
        response_lower = response.lower().strip()

        # Handle various separators - normalize to comma
        for sep in [";", " and ", "\n", "|"]:
            response_lower = response_lower.replace(sep, ",")

        # Extract labels
        predicted = set()
        parts = response_lower.split(",")

        for part in parts:
            part = part.strip()
            # Remove common prefixes/suffixes
            part = re.sub(r"^(the\s+|a\s+|an\s+)", "", part)
            part = re.sub(r"\s*(feature|difference|differs?|aspect)s?\s*$", "", part)

            # Match against valid labels - look for the label word anywhere in the part
            for valid in valid_labels:
                if valid in part:
                    predicted.add(valid)
                    break

        return predicted

    def parse_response(self, response: str, split: str) -> Set[str]:
        """Alias for parse_multi_label_response for backwards compatibility."""
        return self.parse_multi_label_response(response, split)

    def normalize_labels(self, labels: Set[str], split: str) -> Set[str]:
        """
        Normalize a set of labels to valid labels for the split.

        Handles variations like "colours" -> "color", "sizes" -> "size".
        """
        valid_labels = VALID_SYNTHETIC_LABELS if split == "synthetic" else VALID_NATURAL_LABELS

        normalized = set()
        for label in labels:
            label_lower = label.lower().strip()

            # Handle common variations
            variations = {
                "colour": "color",
                "colours": "color",
                "colors": "color",
                "oriented": "orientation",
                "orientations": "orientation",
                "rotated": "orientation",
                "rotation": "orientation",
                "sized": "size",
                "sizes": "size",
                "shaped": "shape",
                "shapes": "shape",
                "focused": "focus",
                "focusing": "focus",
                "locations": "location",
                "positioned": "location",
                "position": "location",
                "patterns": "pattern",
                "patterned": "pattern",
            }

            if label_lower in variations:
                label_lower = variations[label_lower]

            if label_lower in valid_labels:
                normalized.add(label_lower)

        return normalized

    def format_labels(self, labels: Set[str]) -> str:
        """Format a set of labels as a comma-separated, title-cased string."""
        if not labels:
            return ""
        # Sort for consistent output
        sorted_labels = sorted(labels)
        return ", ".join(label.title() for label in sorted_labels)


def get_valid_labels(split: str) -> Set[str]:
    """Get the set of valid labels for a split."""
    return VALID_SYNTHETIC_LABELS.copy() if split == "synthetic" else VALID_NATURAL_LABELS.copy()
