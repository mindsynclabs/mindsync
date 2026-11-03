# MindSync: Enterprise-Grade, Privacy-First AI Platform

MindSync is a lightweight, modern, and high-performance open-source AI platform. Designed with a modular, privacy-first, and local-first architecture, MindSync empowers users and organizations to execute agentic workflows, perform intelligent retrieval-augmented generation (RAG), and continuously fine-tune models—all without compromising data ownership or operational efficiency.

---

## 1. Core Architectural Pillars

* **Privacy-First & Zero-Tracking (Stateless Operating Mode):** MindSync is architected to respect user privacy. It operates in a completely stateless manner on the server side with zero telemetry, zero session-tracking, and no mandatory persistent chat-logging. Llama/Qwen models run locally via Ollama or vLLM, and vector data remains embedded.
* **Pure-Python Native Integration:** Built around **FastAPI** and **PydanticAI**, the execution core eliminates multi-language communication overhead, enabling unified schema validation, type safety, and async-first processing.
* **Modular Engine Paradigm:** Distinct, isolated modules handle execution (Agents), retrieval (Knowledge), execution/inference (Models), custom learning (Training), and routing (Router/Gateway).
* **Open Standard Interoperability:** Implements support for the **Model Context Protocol (MCP)**, allowing agents to seamlessly interact with external systems, tools, and local/remote datasets.

---

## 2. High-Level System Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │           User Interface (UI)           │
                               │        Next.js / Local IndexedDB        │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │             MindSync Router             │
                               │          FastAPI + LiteLLM Proxy        │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │             MindSync Agents             │
                               │               PydanticAI                │
                               └────────┬───────────────────┬────────────┘
                                        │                   │
                     ┌──────────────────┘                   └──────────────────┐
                     ▼                                                         ▼
    ┌───────────────────────────────────┐                     ┌───────────────────────────────────┐
    │         MindSync Knowledge        │                     │          MindSync Models          │
    │     LanceDB (Embedded Vector)     │                     │     Ollama (Dev) / vLLM (Prod)    │
    └───────────────────────────────────┘                     └─────────────────┬─────────────────┘
                                                                                ▲
                                                                                │ (GGUF Export)
                                                              ┌─────────────────┴─────────────────┐
                                                              │          MindSync Training        │
                                                              │          Unsloth Fine-Tuning      │
                                                              └───────────────────────────────────┘
```

---

## 3. Detailed Modules & Key Features

### 3.1 MindSync Agents (Agent Engine)
* **Powered By:** `PydanticAI`
* **Features:**
  * **Strict Type Safety:** Uses Pydantic v2 schemas for rigid input/output validation, preventing structural JSON degradation from LLMs.
  * **Tool Integration & Calling:** Out-of-the-box tool registry for file operations, mathematical computations, and vector searches.
  * **Context & Dependency Injection:** Efficiently injects runtime dependencies (e.g., database clients, auth contexts) per request without global state pollution.
  * **MCP Client Integration:** Connects to any standard Model Context Protocol (MCP) server to dynamically discover and invoke external tools.

### 3.2 MindSync Knowledge (RAG & Retrieval)
* **Powered By:** `LanceDB` (Embedded Vector Database)
* **Features:**
  * **Zero-Infrastructure Vector DB:** Runs in-process via C++/Rust bindings with zero external service setup.
  * **Hybrid Search:** Combines vector similarity search with traditional full-text BM25 search for precise document retrieval.
  * **Local File Ingestion:** Supports automated chunking and embedding for local PDFs, Markdown files, and structured datasets.
  * **Disk-Based Storage:** Stores vectors and disk-backed data efficiently, scaling gracefully on local developer machines or edge nodes.

### 3.3 MindSync Models (Inference Engine)
* **Powered By:** `Ollama` (Development) / `vLLM` (Production) + `LiteLLM` (Cloud Gateway)
* **Features:**
  * **Dual Inference Engines:** 
    * **Development:** Lightweight, one-command local execution using Ollama.
    * **Production:** High-throughput, multi-GPU parallelized inference via vLLM with PagedAttention.
  * **Unified Cloud Abstraction:** Integrates `LiteLLM` to act as a fallback or proxy for cloud LLMs (e.g., OpenAI, Anthropic, DeepSeek) using standard OpenAI-compatible endpoints.
  * **100% Offline Capability:** Operates air-gapped without needing internet access for local model inference.

### 3.4 MindSync Router (Gateway & Middleware)
* **Powered By:** `FastAPI`
* **Features:**
  * **Async Execution:** High-concurrency routing built on Starlette and Uvicorn.
  * **Automated API Documentation:** Provides instant OpenAPI/Swagger documentation at `/docs`.
  * **Stateless Request Handling:** Guarantees zero persistent storage of prompt payloads unless explicitly configured by the user.
  * **Real-Time Streaming:** Supports Server-Sent Events (SSE) and WebSockets for low-latency token streaming to the UI.

### 3.5 MindSync Training (Fine-Tuning Module)
* **Powered By:** `Unsloth`
* **Features:**
  * **Efficient Fine-Tuning:** Leverages LoRA and QLoRA to reduce VRAM consumption by up to 80% while accelerating training speeds by 2–5x.
  * **Local Model Export:** Exports trained weights directly to **GGUF** format, ready to be mounted back into Ollama for localized execution.
  * **Continuous Improvement:** Enables targeted alignment on domain-specific datasets and custom execution traces.

---

## 4. Privacy & Data Governance Specifications

1. **Stateless Processing:**
   * No conversation histories, logs, or prompt telemetry are written to disk or transmitted to remote servers.
2. **Client-Side Persistence:**
   * All session histories, user preferences, and chat threads reside strictly within the client browser (e.g., via `IndexedDB`) or local memory.
3. **Air-Gapped Readiness:**
   * MindSync can run completely disconnected from the public internet when backed by local Ollama/vLLM instances and embedded LanceDB storage.

---

## 5. Technology Stack Summary

| Function | Selected Technology | Deployment Strategy |
| :--- | :--- | :--- |
| **Agent Core** | PydanticAI | Python Package (In-process) |
| **Vector Search** | LanceDB | Embedded / Local File |
| **Local LLM Inference** | Ollama / vLLM | Local Daemon / Production GPU Server |
| **Cloud Proxy** | LiteLLM | Middleware Layer |
| **Backend Framework** | FastAPI | Python Async Server |
| **Fine-Tuning** | Unsloth | PyTorch / CUDA Pipeline |
| **Frontend UI** | Next.js + Tailwind CSS | Static / SSR Web Client |
| **Observability (Opt-in)**| Langfuse | Self-Hosted Open-Source |