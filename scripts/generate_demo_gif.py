"""
Generate an animated GIF showing Local GenAI Stack workflow:
PII Redaction -> Semantic Router -> Hybrid Retrieval -> Token Streaming.
"""

import os

from PIL import Image, ImageDraw

os.makedirs("assets", exist_ok=True)

width, height = 800, 480
bg_color = (15, 23, 42)  # Slate 900
card_bg = (30, 41, 59)  # Slate 800
text_main = (248, 250, 252)
text_muted = (148, 163, 184)
accent_cyan = (56, 189, 248)
accent_green = (74, 222, 128)
accent_amber = (251, 191, 36)
accent_purple = (192, 132, 252)

frames_data = [
    {
        "title": "Step 1: Inbound Request with Sensitive PII",
        "badge": "INGRESS",
        "badge_color": accent_cyan,
        "lines": [
            ("Client -> POST /v1/chat/completions", accent_cyan),
            ("", text_main),
            ("User Input:", text_muted),
            ('"Audit report for EMP-98214 (contact: alice@company.com)', accent_amber),
            (' regarding confidential Project Phoenix 5G architecture."', accent_amber),
            ("", text_main),
            ("Status: Intercepted by Privacy Engine (Fail-Closed Mode Active)", text_muted),
        ],
    },
    {
        "title": "Step 2: Real-time PII & Codename Sanitization",
        "badge": "PRIVACY SHIELD",
        "badge_color": accent_purple,
        "lines": [
            ("Presidio Engine + Custom Regex Entity Recognizers:", accent_purple),
            ("", text_main),
            ("✓ EMPLOYEE_ID     -> [REDACTED: <EMPLOYEE_ID>]", accent_green),
            ("✓ EMAIL_ADDRESS   -> [REDACTED: <EMAIL_ADDRESS>]", accent_green),
            ("✓ PROJECT_CODENAME-> [REDACTED: <PROJECT_CODENAME>]", accent_green),
            ("", text_main),
            ('Sanitized Prompt: "Audit report for <EMPLOYEE_ID> (<EMAIL_ADDRESS>)', text_main),
            (' regarding confidential <PROJECT_CODENAME> 5G architecture."', text_main),
        ],
    },
    {
        "title": "Step 3: Lightweight Semantic Complexity Routing",
        "badge": "ROUTER",
        "badge_color": accent_cyan,
        "lines": [
            ("BGE Embeddings + Threshold Classifier (Latency: ~12.8ms):", accent_cyan),
            ("", text_main),
            ("Query intent: Technical domain query with architectural context", text_muted),
            ("Cosine similarity to reference vectors: 0.884", accent_amber),
            ("Routing Decision: [slm-local] (Phi-3 Mini 4K Instruct)", accent_green),
            ("", text_main),
            ("Cloud fallback bypassed -> On-device execution maintained", accent_green),
        ],
    },
    {
        "title": "Step 4: Hybrid Retrieval with Reciprocal Rank Fusion",
        "badge": "HYBRID RAG",
        "badge_color": accent_amber,
        "lines": [
            ("Query: 5G Core architecture network slicing", accent_amber),
            ("", text_main),
            (
                "1. BM25 Lexical Search  : 5 keyword chunks retrieved (Rank 1: chunk_402)",
                text_muted,
            ),
            (
                "2. BGE Dense Vectors    : 5 semantic chunks retrieved (Rank 1: chunk_402)",
                text_muted,
            ),
            ("3. RRF Fusion (k=60)    : Composite RRF Score = 0.0328", accent_cyan),
            ("", text_main),
            ("Context Grounding: 3 verified chunks injected into prompt", accent_green),
        ],
    },
    {
        "title": "Step 5: SSE Token Streaming & Telemetry",
        "badge": "EGRESS & METRICS",
        "badge_color": accent_green,
        "lines": [
            ("Response Stream (OpenAI SSE protocol):", accent_green),
            ("", text_main),
            ('"According to the 5G Core specification, network slicing allocates', text_main),
            (' dedicated User Plane Functions (UPF) to guarantee SLA throughput..."', text_main),
            ("", text_main),
            ("Prometheus /metrics updated: requests_total +1, pii_entities_total +3", accent_cyan),
            ("Total end-to-end latency: 412ms | Memory footprint: 4.8 GB VRAM", accent_green),
        ],
    },
]

images = []
for data in frames_data:
    im = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(im)

    # Header bar
    draw.rectangle([(20, 20), (width - 20, 70)], fill=card_bg)
    draw.text((40, 32), "LOCAL GENAI STACK — SECURE PRIVACY PIPELINE", fill=text_muted)
    draw.rectangle([(width - 170, 30), (width - 40, 60)], fill=(45, 55, 72))
    draw.text((width - 155, 37), data["badge"], fill=data["badge_color"])

    # Content card
    draw.rectangle([(20, 85), (width - 20, height - 30)], fill=card_bg)
    draw.text((45, 105), data["title"], fill=accent_cyan)
    draw.line([(45, 135), (width - 45, 135)], fill=(51, 65, 85), width=2)

    y = 155
    for line, col in data["lines"]:
        draw.text((50, y), line, fill=col)
        y += 32

    # Repeat each frame for smooth 2-second pacing in GIF
    for _ in range(4):
        images.append(im)

images[0].save("assets/demo.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
print("Demo GIF successfully saved to assets/demo.gif")
