import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Tokenizer loaded.")


# ============================================================
# Configure 4-bit Quantization
# ============================================================

print("Configuring 4-bit quantization...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Load Model
# ============================================================

print("Loading Phi-3 Mini...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)

print("Model loaded successfully!")

print("GPU:", torch.cuda.get_device_name(0))

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)


# ============================================================
# Generate Response Function
# ============================================================

def generate_response(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Convert chat messages into model input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
    )

    # Remove the original prompt from the output
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return response


# ============================================================
# Test Prompts
# ============================================================

prompts = [
    "Explain machine learning.",
    "Explain deep learning.",
    "Explain reinforcement learning.",
    "Explain neural networks.",
    "Explain transformers.",
    "Explain embeddings.",
    "Explain RAG.",
    "Explain vector databases.",
]


# ============================================================
# Run Tests
# ============================================================

print("\n")
print("=" * 70)
print("Starting Phi-3 tests")
print("=" * 70)


for i, prompt in enumerate(prompts, start=1):

    print("\n")
    print("=" * 70)
    print(f"TEST {i}")
    print(f"PROMPT: {prompt}")
    print("=" * 70)

    print("\nGenerating response...\n")

    response = generate_response(prompt)

    print("Phi-3 response:")
    print("-" * 70)
    print(response)


# ============================================================
# Final GPU Memory
# ============================================================

print("\n")
print("=" * 70)
print("Finished all tests")
print("=" * 70)

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)