"""Generate and verify eval/datasets/chunking_qa.jsonl."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))

from rag_service.parsers import extract_pages_from_pdf

docs = {
    "enterprise_rag_sample.pdf": extract_pages_from_pdf(REPO_ROOT / "eval" / "docs" / "enterprise_rag_sample.pdf"),
    "5g_core_architecture.pdf": extract_pages_from_pdf(REPO_ROOT / "eval" / "docs" / "5g_core_architecture.pdf"),
    "cloud_native_telecom_infrastructure.pdf": extract_pages_from_pdf(REPO_ROOT / "eval" / "docs" / "cloud_native_telecom_infrastructure.pdf"),
}

import re

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

doc_texts = {name: normalize_text(" ".join(t for _, t in pages)) for name, pages in docs.items()}

# Curate 36 precise questions with exact verbatim substrings
qa_pairs = [
    # Document 1: enterprise_rag_sample.pdf (8 questions)
    ("What does the Enterprise AI Platform provide?", "deploying and operating artificial intelligence", "enterprise_rag_sample.pdf"),
    ("How can small language models be served on the platform?", "OpenAI-compatible API", "enterprise_rag_sample.pdf"),
    ("What does the serving layer manage?", "manages model inference and accepts chat completion requests", "enterprise_rag_sample.pdf"),
    ("How are documents processed in the ingestion pipeline?", "extracts text, cleans the text, and divides it into smaller chunks", "enterprise_rag_sample.pdf"),
    ("What are document chunks converted into for semantic search?", "converted into vector embeddings", "enterprise_rag_sample.pdf"),
    ("What two components are combined in RAG?", "document retrieval with language model generation", "enterprise_rag_sample.pdf"),
    ("What should incoming user prompts be checked for?", "personally identifiable information", "enterprise_rag_sample.pdf"),
    ("What can detected PII be replaced with?", "anonymized placeholders", "enterprise_rag_sample.pdf"),

    # Document 2: 5g_core_architecture.pdf (14 questions)
    ("What architecture does the 3GPP 5G Core adopt?", "cloud-native Service-Based Architecture", "5g_core_architecture.pdf"),
    ("What protocol and payload format do 5GC control plane NFs use?", "HTTP/2 protocol and JavaScript Object Notation", "5g_core_architecture.pdf"),
    ("Which interface connects the RAN control plane to the AMF?", "N2 interface", "5g_core_architecture.pdf"),
    ("What signaling protocol is terminated by the AMF on the N1 interface?", "Non-Access Stratum", "5g_core_architecture.pdf"),
    ("What responsibilities belong to the AMF?", "registration management, connection management", "5g_core_architecture.pdf"),
    ("Which network function selects and controls the UPF?", "Session Management Function", "5g_core_architecture.pdf"),
    ("What is the role of the Network Repository Function (NRF)?", "catalog of all active NF instances", "5g_core_architecture.pdf"),
    ("How does the NSSF choose slice instances for users?", "subscribed S-NSSAI", "5g_core_architecture.pdf"),
    ("What is the primary data path anchor in 5G networks?", "User Plane Function (UPF) is the primary data path anchor", "5g_core_architecture.pdf"),
    ("Which rules are used by the UPF for packet classification?", "Packet Detection Rules (PDR)", "5g_core_architecture.pdf"),
    ("Which interface connects the UPF to external Data Networks?", "N6 interface", "5g_core_architecture.pdf"),
    ("Which function handles billing usage reporting from the UPF?", "Charging Function (CHF)", "5g_core_architecture.pdf"),
    ("What performance characteristics define SST 1?", "Enhanced Mobile Broadband (eMBB)", "5g_core_architecture.pdf"),
    ("What latency requirement defines SST 2 URLLC slices?", "sub-millisecond radio transit time", "5g_core_architecture.pdf"),

    # Document 3: cloud_native_telecom_infrastructure.pdf (14 questions)
    ("What are operators transitioning to from Virtual Machines?", "Containerized Network Functions (CNFs)", "cloud_native_telecom_infrastructure.pdf"),
    ("What demands distinguish telecom user plane workloads?", "deterministic low latency and extreme packet throughput", "cloud_native_telecom_infrastructure.pdf"),
    ("How does SR-IOV present a physical PCIe NIC to containers?", "multiple virtual devices (VFs)", "cloud_native_telecom_infrastructure.pdf"),
    ("How does DPDK eliminate kernel interrupt overhead?", "user-space poll-mode drivers", "cloud_native_telecom_infrastructure.pdf"),
    ("At what layer does XDP provide in-kernel packet filtering?", "device driver layer before socket buffers are allocated", "cloud_native_telecom_infrastructure.pdf"),
    ("Why do standard Kubernetes CNI plugins fall short in telco pods?", "require multiple network interfaces", "cloud_native_telecom_infrastructure.pdf"),
    ("What meta-plugin attaches multiple network interfaces to a pod?", "Multus CNI acts as a meta-plugin", "cloud_native_telecom_infrastructure.pdf"),
    ("What device plugins back secondary pod interfaces?", "Macvlan, IPvlan, or SR-IOV device plugins", "cloud_native_telecom_infrastructure.pdf"),
    ("What security architecture governs telecom cloud infrastructure?", "Zero-Trust Architecture (ZTA)", "cloud_native_telecom_infrastructure.pdf"),
    ("How is inter-service communication encrypted in telco cloud?", "mutual Transport Layer Security (mTLS)", "cloud_native_telecom_infrastructure.pdf"),
    ("Which agents issue cryptographic identity certificates?", "SPIFFE/SPIRE agents", "cloud_native_telecom_infrastructure.pdf"),
    ("How are metrics gathered in telco cloud observability?", "Prometheus pull endpoints", "cloud_native_telecom_infrastructure.pdf"),
    ("How is distributed tracing propagated across queues?", "W3C TraceContext propagation", "cloud_native_telecom_infrastructure.pdf"),
    ("What probes provide real-time packet telemetry for anomaly detection?", "eBPF probes for anomaly detection", "cloud_native_telecom_infrastructure.pdf"),
]

# Verify all substrings exist
for q, sub, doc in qa_pairs:
    text = doc_texts[doc]
    if sub not in text:
        raise ValueError(f'Substring "{sub}" not found in {doc}')

print(f"Verification passed! All {len(qa_pairs)} substrings exist exactly in the documents.")

out_path = REPO_ROOT / "eval" / "datasets" / "chunking_qa.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for q, sub, doc in qa_pairs:
        f.write(json.dumps({"question": q, "expected_substring": sub, "doc_name": doc}) + "\n")

print(f"Wrote {len(qa_pairs)} verified records to {out_path}")
