"""Generate clean, rich, multi-page technical PDFs for evaluation."""

from pathlib import Path
import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "eval" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def create_pdf(path: Path, pages: list[str]):
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, page_text.strip(), fontsize=11, fontname="helv", lineheight=1.3)
    doc.save(path)
    doc.close()
    print(f"Created {path}")

doc2_pages = [
    """5G Core Network Architecture Overview

Section 1. Introduction to Service-Based Architecture
The 3GPP 5G Core (5GC) network adopts a cloud-native Service-Based Architecture (SBA). Control plane Network Functions (NFs) communicate with each other over standardized Service-Based Interfaces (SBI) using HTTP/2 protocol and JavaScript Object Notation (JSON) payloads. The control plane is decoupled from the user plane, allowing independent scaling and distributed deployment.

Section 2. Control Plane Network Functions
The Access and Mobility Management Function (AMF) terminates the Radio Access Network (RAN) control plane on the N2 interface and the Non-Access Stratum (NAS) signaling on the N1 interface. The AMF handles registration management, connection management, reachability management, and mobility management of User Equipment (UE).

The Session Management Function (SMF) handles session establishment, modification, and release. It selects and controls the User Plane Function (UPF) for data routing and allocates IP addresses to user devices.

The Network Repository Function (NRF) provides service registration and discovery, acting as the catalog of all active NF instances.

The Network Slice Selection Function (NSSF) selects the optimal network slice instance for registered users based on subscribed S-NSSAI.""",
    """Section 3. User Plane Architecture and UPF Operations
The User Plane Function (UPF) is the primary data path anchor in 5G networks, handling packet routing, forwarding, and Quality of Service (QoS) enforcement.

Key responsibilities of the UPF include:
1. Packet inspection and classification according to Packet Detection Rules (PDR).
2. Traffic forwarding and routing towards external Data Networks (DN) over the N6 interface.
3. Lawful interception and usage reporting for billing through the Charging Function (CHF).
4. Uplink and downlink transport level packet marking using DiffServ (DSCP).
5. Serving as the PDU session anchor point for seamless mobility across gNodeB base stations.

Section 4. Network Slicing and Policy Control
Network slicing partitions physical network resources into dedicated virtual end-to-end networks. Each slice is characterized by a Slice/Service Type (SST) and an optional Slice Differentiator (SD).

Standard SST values defined by 3GPP include:
- SST 1: Enhanced Mobile Broadband (eMBB) requiring high data rates and spectral efficiency.
- SST 2: Ultra-Reliable Low-Latency Communication (URLLC) with sub-millisecond radio transit time.
- SST 3: Massive Machine Type Communication (mMTC) supporting high connection density for IoT sensors.

The Policy Control Function (PCF) provides unified policy rules to control plane functions and governs dynamic QoS."""
]

doc3_pages = [
    """Cloud-Native Telecommunications Infrastructure

Chapter 1. Containerized Network Functions (CNFs)
Modern telecommunications operators are transitioning from Virtual Machines (VNFs) to Containerized Network Functions (CNFs). CNFs leverage microservices architectures orchestrated by Kubernetes, enabling automated lifecycle management, zero-downtime rolling deployments, and optimized compute footprint.

Unlike traditional enterprise workloads, telecom user plane workloads demand deterministic low latency and extreme packet throughput.

Chapter 2. High-Performance Networking Acceleration
To achieve wire-speed processing in containerized environments, CNFs utilize specialized kernel-bypass acceleration technologies:
1. Single Root I/O Virtualization (SR-IOV): Allows a single physical PCIe network interface card (NIC) to appear as multiple virtual devices (VFs), which can be passed directly into container pods.
2. Data Plane Development Kit (DPDK): Provides user-space poll-mode drivers that bypass the Linux kernel network stack, eliminating interrupt processing and context-switching overhead.
3. Express Data Path (XDP) and eBPF: Enables programmable in-kernel packet filtering and steering at the device driver layer before socket buffers are allocated.""",
    """Chapter 3. Kubernetes CNI Plugins in Telco Deployments
Standard Kubernetes clusters provide a single network interface per pod using a Container Network Interface (CNI) plugin like Flannel or Calico. However, telecom CNFs require multiple network interfaces: one for cluster management and control plane traffic, and one or more high-performance interfaces for user plane packet forwarding.

Multus CNI acts as a meta-plugin that attaches multiple network interfaces to a single Kubernetes pod. Secondary interfaces can be backed by Macvlan, IPvlan, or SR-IOV device plugins.

Chapter 4. Zero-Trust Security and Observability
Telecom cloud infrastructure operates under a Zero-Trust Architecture (ZTA). Inter-service communication is secured via mutual Transport Layer Security (mTLS) with cryptographic identity certificates issued by SPIFFE/SPIRE agents.

Observability is provided through OpenTelemetry standards:
- Metrics collection via Prometheus pull endpoints.
- Distributed tracing using W3C TraceContext propagation across asynchronous message queues.
- Real-time packet telemetry collected via eBPF probes for anomaly detection."""
]

doc1_pages = [
    """Enterprise AI Platform — Technical Overview

Section 1. Platform Capabilities
The Enterprise AI Platform provides services for deploying and operating artificial intelligence applications. The platform supports model serving, document processing, retrieval, and API-based access to language models.

Section 2. Model Serving Layer
Small language models can be served through an OpenAI-compatible API. The serving layer manages model inference and accepts chat completion requests with low latency.

Section 3. Document Ingestion Pipeline
How are documents processed in the ingestion pipeline? The pipeline extracts text, cleans the text, and divides it into smaller chunks. Document chunks are converted into vector embeddings for semantic search. Retrieval-Augmented Generation (RAG) combines document retrieval with language model generation.

Section 4. Privacy and Guardrails
Incoming user prompts are checked for personally identifiable information (PII). Detected PII entities like email addresses, phone numbers, employee IDs, and internal codenames are replaced with anonymized placeholders before model inference."""
]

doc4_pages = [
    """Distributed Consensus Architecture & Raft Protocol Specification

Section 1. Core State Machine Replication & Node Roles
In distributed state machine replication, consensus guarantees that a cluster of replicated state machines produces identical execution sequences despite network partitions and arbitrary node crashes (fail-stop model). The Raft protocol divides time into numbered terms of arbitrary duration, acting as a logical clock to detect stale state.

Each cluster node operates in one of three distinct roles:
1. Follower: Passive responder to Remote Procedure Calls (RPCs). If no communication is received within the randomized election timeout window, the follower transitions to candidate state.
2. Candidate: Initiates leader election by incrementing currentTerm, voting for self, and broadcasting RequestVote RPCs to all peers.
3. Leader: Manages all client write proposals, appends entries to its local Write-Ahead Log (WAL), and drives log replication across quorum peers.

Cluster Timing Invariants:
- Heartbeat Interval: Default is 50 ms (must be substantially smaller than broadcast time).
- Election Timeout: Randomized window between 150 ms and 300 ms to break split-vote ties.
- Client Port: 2379 for client requests; Peer Port: 2380 for inter-node raft replication.""",

    """Section 2. Log Replication Protocol & Quorum Invariants

When the leader receives a state transition command from a client, it assigns a monotonic log index and the current term number. The entry is appended to its local log and broadcast via AppendEntries RPCs.

AppendEntries RPC Structure and Fields:
- term: Leader's current term number.
- leaderId: Identifier so followers can redirect client connections.
- prevLogIndex: Index of log entry immediately preceding new entries.
- prevLogTerm: Term of prevLogIndex entry.
- entries[]: Array of log entries to store (empty for periodic heartbeats).
- leaderCommit: Leader's current commitIndex.

Log Matching Property & Invariants:
1. If two entries in different logs have the exact same index and term, they store the exact same command.
2. If two entries in different logs have the same index and term, then their logs are identical in all preceding entries.

Error Codes and Handling:
- ERR_TERM_OUTDATED (Code 4010): Returned when a sender's term is lower than the receiver's currentTerm; the sender immediately steps down to follower.
- ERR_LOG_DIVERGENCE (Code 4020): Returned when the receiver's log does not contain an entry matching prevLogIndex and prevLogTerm; the leader decrements nextIndex and retries.""",

    """Section 3. Joint Consensus, Log Compaction, and Snapshotting

Cluster Membership Changes:
Dynamic configuration changes (adding or removing nodes) employ a two-phase Joint Consensus approach. During transition, decisions require independent majorities from both the old configuration (C_old) and the new configuration (C_new). Once the joint consensus entry is committed, the leader commits the final C_new configuration.

Log Compaction and Storage Engine:
Unbounded log growth exhausts memory and disk storage. Raft utilizes asynchronous snapshotting:
1. State Machine Snapshots: Periodic checkpoint of the entire deterministic state machine serialized to disk.
2. Compaction Threshold: Default triggers when WAL exceeds 10,000 uncompacted entries or 64 MB.
3. InstallSnapshot RPC: Invoked when a lagging follower's nextIndex falls behind the leader's oldest compacted WAL entry.

Fault Tolerance Guarantees:
A Raft cluster of N nodes maintains availability and linearizable read/write consistency under up to F = floor((N - 1) / 2) concurrent failures, verified against Jepsen network partition test suites."""
]

create_pdf(DOCS_DIR / "enterprise_rag_sample.pdf", doc1_pages)
create_pdf(DOCS_DIR / "5g_core_architecture.pdf", doc2_pages)
create_pdf(DOCS_DIR / "cloud_native_telecom_infrastructure.pdf", doc3_pages)
create_pdf(DOCS_DIR / "distributed_consensus_raft_spec.pdf", doc4_pages)
