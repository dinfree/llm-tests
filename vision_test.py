#!/usr/bin/env python3
"""
Simple vision model test script for OpenAI-compatible endpoints.
- BASE_URL is intentionally hardcoded in code.
- Sends one image + text prompt to /chat/completions.
"""

import base64
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Run without .env loading when python-dotenv is unavailable.
    pass

# Requested style: hardcoded base_url in code.
BASE_URL = "http://169.254.63.162:9090/v1"
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "ktransformers"
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "sample_image.jpg"
DEFAULT_PROMPT = "Describe this image in 2-3 sentences."


def fetch_first_model() -> str:
    resp = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError("No models found from /models")
    return data[0]["id"]


def build_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    ext = image_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def run_vision_test(image_path: Path, prompt: str) -> None:
    model_name = fetch_first_model()
    image_data_url = build_data_url(image_path)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "stream": False,
    }

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if resp.status_code != 200:
        print(f"Request failed: HTTP {resp.status_code}")
        print(resp.text[:1000])
        return

    data = resp.json()
    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = str(data)

    print("=" * 72)
    print(f"BASE_URL : {BASE_URL}")
    print(f"MODEL    : {model_name}")
    print(f"IMAGE    : {image_path}")
    print("=" * 72)
    print(text)


if __name__ == "__main__":
    image_arg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    prompt_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT
    run_vision_test(image_arg, prompt_arg)
