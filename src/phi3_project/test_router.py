from .router import IntentRouter


router = IntentRouter()

test_cases = [
    # General
    ("What is the capital of Germany?", "general"),
    ("Who discovered electricity?", "general"),
    ("Explain the water cycle.", "general"),
    ("What is the tallest mountain?", "general"),

    # Technical
    ("How do I create a Python function?", "technical"),
    ("How does an HTTP API work?", "technical"),
    ("Explain SQL indexing.", "technical"),
    ("How do I deploy a Docker container?", "technical"),

    # RAG
    ("What does the uploaded PDF say about security?", "rag"),
    ("Summarize the company document.", "rag"),
    ("According to the uploaded report, what is the deadline?", "rag"),
    ("Find the leave policy in the document.", "rag"),
]


correct = 0

for query, expected in test_cases:
    result = router.classify(query)

    predicted = result["intent"]

    if predicted == expected:
        correct += 1
        status = "PASS"
    else:
        status = "FAIL"

    print(
        f"{status} | "
        f"Expected: {expected:<9} | "
        f"Predicted: {predicted:<9} | "
        f"Score: {result['confidence']:.3f} | "
        f"{query}"
    )


accuracy = correct / len(test_cases)

print()
print(f"Accuracy: {correct}/{len(test_cases)} = {accuracy:.2%}")