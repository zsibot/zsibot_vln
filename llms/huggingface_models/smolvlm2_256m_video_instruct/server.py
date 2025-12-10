import base64
import tempfile
import time
from io import BytesIO
from typing import Any, Dict, List

import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# ------------- Model config -------------

MODEL_PATH = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
).to(DEVICE)

app = FastAPI()


# ------------- OpenAI-like schema -------------

class ChatCompletionMessage(BaseModel):
    role: str
    content: Any  # can be string or list[dict]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]


# ------------- Helpers -------------

def decode_image_from_image_url(image_url: str) -> str:
    """
    Accepts data URL like 'data:image/png;base64,...' and saves to a temp file.
    Returns the temp file path that SmolVLM processor can load.
    """
    if image_url.startswith("data:"):
        # data:image/png;base64,XXXX
        _, b64data = image_url.split(",", 1)
    else:
        raise ValueError("Only base64 data URLs are supported for now.")

    img_bytes = base64.b64decode(b64data)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(img_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def build_conversation_from_request(req: ChatCompletionRequest) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages into the multimodal conversation format
    expected by SmolVLM's processor.apply_chat_template.

    Supports:
      - Text-only: content is a string
      - Vision: content is a list of {type: "text" | "image_url", ...}
    """
    # Take the last user message (simple assumption)
    user_msg = None
    for m in reversed(req.messages):
        if m.role == "user":
            user_msg = m
            break
    if user_msg is None:
        raise ValueError("No user message found")

    contents = user_msg.content

    # ---------- CASE 1: plain string (text-only) ----------
    if isinstance(contents, str):
        text = contents
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        return conversation

    # ---------- CASE 2: list of parts (possibly with image) ----------
    if isinstance(contents, list):
        text_parts: List[str] = []
        image_paths: List[str] = []

        for part in contents:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                img_url = part.get("image_url")
                if img_url is None:
                    continue
                img_path = decode_image_from_image_url(img_url)
                image_paths.append(img_path)

        text = "\n".join(t for t in text_parts if t.strip())

        # No image → treat as text-only
        if len(image_paths) == 0:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
            return conversation

        # For now assume exactly 1 image
        if len(image_paths) != 1:
            raise ValueError(f"Expected 0 or 1 image, got {len(image_paths)}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_paths[0]},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return conversation

    # Anything else is unsupported
    raise ValueError(f"Unsupported content format: {type(contents)}")


# ------------- Main endpoint -------------

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    # Build SmolVLM conversation
    conversation = build_conversation_from_request(req)

    # Turn into model inputs
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE, dtype=torch.bfloat16)

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
        )

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    answer: str = generated_texts[0]

    # IMPORTANT: return `content` as a plain string (for your existing code)
    now = int(time.time())
    response: Dict[str, Any] = {
        "id": "chatcmpl-smolvlm-1",
        "object": "chat.completion",
        "created": now,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,  # <--- string, NOT list-of-parts
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return response


# ------------- Entrypoint -------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
