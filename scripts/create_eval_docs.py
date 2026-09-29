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

create_pdf(DOCS_DIR / "ericsson_5g_core_architecture.pdf", doc2_pages)
create_pdf(DOCS_DIR / "cloud_native_telecom_infrastructure.pdf", doc3_pages)
