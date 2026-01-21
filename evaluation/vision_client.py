"""Unified vision API client supporting Ollama and OpenRouter."""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import aiohttp
from PIL import Image

from .config import EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class VisionResponse:
    """Response from a vision API call."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class VisionClient:
    """Unified client for Ollama and OpenRouter vision APIs."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _encode_image(self, image: Union[str, Path, Image.Image]) -> tuple:
        """
        Encode an image to base64.

        Args:
            image: Can be a file path (str/Path) or PIL Image

        Returns:
            Tuple of (base64_data, mime_type)
        """
        if isinstance(image, (str, Path)):
            # File path
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image}")

            mime_type = self._get_mime_type(str(path))
            with open(path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            return b64_data, mime_type

        elif isinstance(image, Image.Image):
            # PIL Image - convert to PNG bytes
            buffer = io.BytesIO()
            # Convert to RGB if necessary (e.g., RGBA or P mode)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffer, format="PNG")
            b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return b64_data, "image/png"

        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from image path."""
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "image/jpeg")

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.backend == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/salbench"
            headers["X-Title"] = "SalBench Evaluation"
        return headers

    def _build_messages(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages array for the API request."""
        messages = []

        # Add few-shot examples if provided
        if few_shot_examples:
            for example in few_shot_examples:
                example_b64, example_mime = self._encode_image(example["image"])
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{example_mime};base64,{example_b64}"},
                        },
                        {"type": "text", "text": example["question"]},
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": example["answer"],
                })

        # Add the actual query
        image_b64, image_mime = self._encode_image(image)

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        return messages

    async def send_request(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> VisionResponse:
        """
        Send a vision request to the API.

        Args:
            image: Image as file path (str/Path) or PIL Image
            prompt: Text prompt
            few_shot_examples: Optional list of few-shot examples with 'image' and 'question'/'answer' keys

        Returns:
            VisionResponse with the model's response
        """
        await self._ensure_session()

        messages = self._build_messages(image, prompt, few_shot_examples)
        headers = self._build_headers()

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        for attempt in range(self.config.retry_attempts):
            try:
                async with self._session.post(
                    self.config.api_endpoint,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return VisionResponse(
                            content=data["choices"][0]["message"]["content"],
                            model=data.get("model", self.config.model_name),
                            usage=data.get("usage"),
                        )
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"API error (attempt {attempt + 1}/{self.config.retry_attempts}): "
                            f"{response.status} - {error_text}"
                        )
                        if attempt == self.config.retry_attempts - 1:
                            return VisionResponse(
                                content="",
                                model=self.config.model_name,
                                error=f"API error {response.status}: {error_text}",
                            )

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout (attempt {attempt + 1}/{self.config.retry_attempts})"
                )
                if attempt == self.config.retry_attempts - 1:
                    return VisionResponse(
                        content="",
                        model=self.config.model_name,
                        error="Request timed out",
                    )

            except (KeyboardInterrupt, asyncio.CancelledError):
                # Re-raise interrupts immediately so user can cancel
                raise

            except Exception as e:
                logger.warning(
                    f"Request error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )
                if attempt == self.config.retry_attempts - 1:
                    return VisionResponse(
                        content="",
                        model=self.config.model_name,
                        error=str(e),
                    )

            # Exponential backoff
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)

        return VisionResponse(
            content="",
            model=self.config.model_name,
            error="Max retries exceeded",
        )
