import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import httpx

# ---- Load the image and convert to base64 ----
image = Image.open("./bed.jpg").convert("RGB")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---- Create OpenAI-compatible client ----
client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1",   # <-- Your server
    http_client=httpx.Client(trust_env=False)
)

# ---- Send the request ----
resp = client.chat.completions.create(
    model="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + image_b64
                }
            ]
        }
    ],
)

# ---- Print the result ----
print("RESPONSE:")
print(resp.choices[0].message)
