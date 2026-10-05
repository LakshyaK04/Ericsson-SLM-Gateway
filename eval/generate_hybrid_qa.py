"""Generate and verify eval/datasets/hybrid_eval.jsonl with exact substring ground truth."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))

from rag_service.parsers import extract_pages_from_pdf

docs = {
    "enterprise_rag_sample.pdf": extract_pages_from_pdf(
        REPO_ROOT / "eval" / "docs" / "enterprise_rag_sample.pdf"
    ),
    "5g_core_architecture.pdf": extract_pages_from_pdf(
        REPO_ROOT / "eval" / "docs" / "5g_core_architecture.pdf"
    ),
    "cloud_native_telecom_infrastructure.pdf": extract_pages_from_pdf(
        REPO_ROOT / "eval" / "docs" / "cloud_native_telecom_infrastructure.pdf"
    ),
    "distributed_consensus_raft_spec.pdf": extract_pages_from_pdf(
        REPO_ROOT / "eval" / "docs" / "distributed_consensus_raft_spec.pdf"
    ),
}

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


doc_texts = {name: normalize_text(" ".join(t for _, t in pages)) for name, pages in docs.items()}

# Curate 24 questions: 12 keyword/acronym specific + 12 conceptual paraphrase
qa_items = [
    # --- Category A: Keyword / Acronym / Specific Identifier (BM25's strength) ---
    {
        "query": "What port is used for peer-to-peer raft replication?",
        "expected_substring": "Peer Port: 2380 for inter-node raft replication",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "Which error code corresponds to ERR_LOG_DIVERGENCE?",
        "expected_substring": "ERR_LOG_DIVERGENCE (Code 4020)",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "What action occurs when ERR_TERM_OUTDATED is returned?",
        "expected_substring": "ERR_TERM_OUTDATED (Code 4010)",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "When is InstallSnapshot RPC invoked by the leader?",
        "expected_substring": "InstallSnapshot RPC: Invoked when a lagging follower",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "Which interface connects UPF to external Data Networks?",
        "expected_substring": "N6 interface",
        "doc_name": "5g_core_architecture.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "What signaling protocol is terminated by AMF on the N1 interface?",
        "expected_substring": "Non-Access Stratum (NAS) signaling on the N1 interface",
        "doc_name": "5g_core_architecture.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "What rules does UPF use for packet classification?",
        "expected_substring": "Packet Detection Rules (PDR)",
        "doc_name": "5g_core_architecture.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "What Slice/Service Type defines ultra-reliable low latency?",
        "expected_substring": "SST 2: Ultra-Reliable Low-Latency Communication",
        "doc_name": "5g_core_architecture.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "Which technology allows a single PCIe NIC to appear as multiple virtual devices?",
        "expected_substring": "Single Root I/O Virtualization (SR-IOV)",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "What framework provides user-space poll-mode drivers to bypass Linux kernel networking?",
        "expected_substring": "Data Plane Development Kit (DPDK)",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "Which meta-plugin attaches multiple network interfaces to a single Kubernetes pod?",
        "expected_substring": "Multus CNI acts as a meta-plugin",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "keyword_acronym",
    },
    {
        "query": "Which cryptographic identity agents issue certificates for mutual TLS?",
        "expected_substring": "SPIFFE/SPIRE agents",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "keyword_acronym",
    },
    # --- Category B: Conceptual / Paraphrase (Dense's strength) ---
    {
        "query": "How do distributed cluster nodes resolve simultaneous candidacy deadlocks?",
        "expected_substring": "Randomized window between 150 ms and 300 ms to break split-vote ties",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "How does the protocol guarantee that committed log entries will never be superseded?",
        "expected_substring": "Log Matching Property & Invariants",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "What strategy allows safely reconfiguring cluster topology without stopping operations?",
        "expected_substring": "two-phase Joint Consensus approach",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "How is infinite disk exhaustion prevented in the transaction log?",
        "expected_substring": "Unbounded log growth exhausts memory and disk storage. Raft utilizes asynchronous snapshotting",
        "doc_name": "distributed_consensus_raft_spec.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "What architecture pattern separates control signaling from user data traffic in modern mobile cores?",
        "expected_substring": "control plane is decoupled from the user plane",
        "doc_name": "5g_core_architecture.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "How does the core network isolate physical infrastructure into customized logical partitions for customers?",
        "expected_substring": "Network slicing partitions physical network resources into dedicated virtual end-to-end networks",
        "doc_name": "5g_core_architecture.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "How can packet filtering be executed early in the operating system driver before socket memory is allocated?",
        "expected_substring": "device driver layer before socket buffers are allocated",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "Why are telecommunications software deployments moving away from heavy virtual machine hypervisors?",
        "expected_substring": "transitioning from Virtual Machines (VNFs) to Containerized Network Functions (CNFs)",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "How do distributed trace contexts follow requests across non-blocking message buses?",
        "expected_substring": "W3C TraceContext propagation across asynchronous message queues",
        "doc_name": "cloud_native_telecom_infrastructure.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "What mechanism guarantees that confidential user identifiers do not reach third-party language models?",
        "expected_substring": "Detected PII entities like email addresses, phone numbers, employee IDs, and internal codenames are replaced with anonymized placeholders",
        "doc_name": "enterprise_rag_sample.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "How are raw technical documents transformed into indexable semantic search units?",
        "expected_substring": "extracts text, cleans the text, and divides it into smaller chunks. Document chunks are converted into vector embeddings",
        "doc_name": "enterprise_rag_sample.pdf",
        "category": "conceptual_paraphrase",
    },
    {
        "query": "What protocol allows external client applications to query local language models seamlessly?",
        "expected_substring": "Small language models can be served through an OpenAI-compatible API",
        "doc_name": "enterprise_rag_sample.pdf",
        "category": "conceptual_paraphrase",
    },
]

# Verify all substrings exist
for item in qa_items:
    doc_name = item["doc_name"]
    sub = item["expected_substring"]
    assert sub in doc_texts[doc_name], f"Missing substring in {doc_name}: {sub}"

out_path = REPO_ROOT / "eval" / "datasets" / "hybrid_eval.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for item in qa_items:
        f.write(json.dumps(item) + "\n")

print(f"Verified and saved {len(qa_items)} test queries to {out_path}")
