from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "./bed.jpg"},          # ← use path here
            {"type": "text", "text": "Can you describe this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
    )

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])
