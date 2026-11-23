"""Every Cloudflare Workers AI model the app uses: chat and handwriting OCR (Qwen), and lesson embeddings (bge-m3)."""

import base64
import os
from typing import Any

import requests
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI

from tools import arabic_letters_only

DEFAULT_CHAT_MODEL = "@cf/qwen/qwen3.8-27b"
DEFAULT_VISION_MODEL = "@cf/qwen/qwen3.8-27b"
DEFAULT_EMBED_MODEL = "@cf/baai/bge-m3"
EMBED_BATCH_SIZE = 50

OCR_PROMPT = (
    "This image is a learner's handwriting of one Arabic letter, syllable or word, drawn with a mouse or finger. "
    "Transcribe exactly the Arabic letters that are written, in order, without harakat. "
    "Look carefully at the number and position of dots, and at hamza marks and their seats (أ إ ؤ ئ ء), "
    "and tell ة from ه and ى from ي. Do not correct the spelling and do not guess a dictionary word: "
    "if a letter is written wrongly, transcribe it as written. Output only the Arabic letters, nothing else."
)


# Qwen request options: reasoning off by default (fast, prompts carry the lesson); CF_CHAT_REASONING=low|medium turns it on.
def chat_options(model_name: str) -> dict[str, Any]:
    if "qwen" not in model_name:
        return {}
    effort = (os.environ.get("CF_CHAT_REASONING") or "off").strip().lower()
    if effort in ("low", "medium", "xhigh"):
        return {"reasoning_effort": effort}
    return {"chat_template_kwargs": {"enable_thinking": False}}


class Models:
    # Reads Cloudflare settings from the environment and builds the chat model.
    def __init__(self) -> None:
        self.account_id = os.environ.get("CF_ACCOUNT_ID")
        self.api_token = os.environ.get("CF_AI_API_TOKEN") or os.environ.get("CF_API_TOKEN")
        self.vision_model = os.environ.get("CF_VISION_MODEL") or DEFAULT_VISION_MODEL
        self.embed_model = os.environ.get("CF_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.chat = ChatCloudflareWorkersAI(
            account_id=os.environ.get("CF_ACCOUNT_ID"),
            api_token=os.environ.get("CF_AI_API_TOKEN") or os.environ.get("CF_API_TOKEN"),
            model=os.environ.get("CF_MODEL") or DEFAULT_CHAT_MODEL,
            model_kwargs=chat_options(os.environ.get("CF_MODEL") or DEFAULT_CHAT_MODEL),
        )

    # Reads the handwritten letters in a PNG and returns Arabic letters only, harakat stripped.
    def read_handwriting(self, png_bytes: bytes) -> str:
        if "qwen" in self.vision_model:
            raw_text = self._read_with_qwen(png_bytes)
        elif "moondream" in self.vision_model:
            raw_text = self._read_with_moondream(png_bytes)
        else:
            raw_text = self._read_with_llama_vision(png_bytes)
        return arabic_letters_only(raw_text)

    # Embeds texts with bge-m3 in batches and returns one 1024-number vector per text, in order.
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            result = self._run_model(self.embed_model, {"text": batch})
            data = result.get("data") or []
            if len(data) != len(batch):
                raise RuntimeError(f"Embedding model returned {len(data)} vectors for {len(batch)} texts")
            vectors.extend(data)
        return vectors

    # Sends the one-time {"prompt": "agree"} request that accepts Meta's Llama license (Llama vision only).
    def accept_vision_license(self) -> dict[str, Any]:
        return self._run_vision({"prompt": "agree"})

    # Qwen: OpenAI-style message with an image_url part; thinking off for speed.
    def _read_with_qwen(self, png_bytes: bytes) -> str:
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        content = [{"type": "text", "text": OCR_PROMPT}, {"type": "image_url", "image_url": {"url": data_url}}]
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 200, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        choices = self._run_vision(body).get("choices") or [{}]
        return choices[0].get("message", {}).get("content") or ""

    # Llama 3.2 Vision: image goes in the documented uint8-array "image" field.
    def _read_with_llama_vision(self, png_bytes: bytes) -> str:
        body = {"prompt": OCR_PROMPT, "image": list(png_bytes), "max_tokens": 64, "temperature": 0}
        return self._run_vision(body).get("response") or ""

    # Moondream: image goes in as a base64 data URI; the text comes back in "answer".
    def _read_with_moondream(self, png_bytes: bytes) -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        body = {"task": "query", "image": data_uri, "question": OCR_PROMPT,
                "reasoning": False, "max_tokens": 64, "temperature": 0}
        return self._run_vision(body).get("answer") or ""

    # POSTs to the Workers AI REST endpoint for the vision model and returns its "result".
    def _run_vision(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._run_model(self.vision_model, body)

    # POSTs a body to any Workers AI model and returns "result"; raises a clear error when success is false.
    def _run_model(self, model_name: str, body: dict[str, Any]) -> dict[str, Any]:
        if not self.account_id or not self.api_token:
            raise RuntimeError("Set CF_ACCOUNT_ID and CF_AI_API_TOKEN in .env")
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model_name}"
        response = requests.post(url, headers={"Authorization": f"Bearer {self.api_token}"}, json=body, timeout=90)
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = {}
        if not response.ok or not payload.get("success", False):
            raise RuntimeError(f"Cloudflare call to {model_name} failed ({response.status_code}): {payload.get('errors')}")
        return payload.get("result") or {}
