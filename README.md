# MindSync 🧠⚡

**MindSync** is an enterprise-grade modular AI platform designed to unify Retrieval-Augmented Generation (RAG), dynamic model orchestration, automated fine-tuning, and multi-agent workflows under a single cohesive system architecture.

---

## 🏛️ System Architecture & Modules

MindSync is divided into five core operational modules:

```
                  ┌──────────────────────┐
                  │    User Interface    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   MindSync Router    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   MindSync Agents    │
                  └─────┬──────────┬─────┘
                        │          │
         ┌──────────────┘          └──────────────┐
         ▼                                        ▼
┌──────────────────┐                    ┌──────────────────┐
│ MindSync Knowledge│                   │  MindSync Models │
└──────────────────┘                    └────────┬─────────┘
                                                 ▲
                                                 │
                                        ┌────────┴─────────┐
                                        │MindSync Training │
                                        └──────────────────┘
```

| Module | Scope & Function |
| :--- | :--- |
| **`MindSync Knowledge`** | Enterprise RAG pipeline supporting hybrid search (Dense Vector + Sparse BM25), semantic chunking, RBAC document access, and automated data ingestion. |
| **`MindSync Models`** | Model abstraction tier facilitating unified API calls across cloud endpoints (OpenAI, Anthropic, Google) and self-hosted instances (vLLM, Ollama). |
| **`MindSync Training`** | Dataset curator and PEFT/LoRA fine-tuning workspace designed to train domain-specific models from agent execution logs. |
| **`MindSync Router`** | Real-time routing gateway evaluating intent, latency, cost constraints, and security policy to dispatch tasks to appropriate agents and models. |
| **`MindSync Agents`** | Multi-agent execution engine powered by state graph workflows, supporting tool integration, persistent memory, and automated plan-and-execute logic. |

---

## 🚀 Key Features

* **Hybrid Retrieval System:** Combines vector embeddings with keyword-based retrieval for high accuracy across complex enterprise documentation.
* **Cost & Latency Intelligent Routing:** Dynamically selects high-reasoning models (e.g., Claude 3.5 Sonnet / GPT-4o) for complex tasks while routing simple text parsing to lighter, fast local models (e.g., Llama 3 8B).
* **Autonomous Multi-Agent Orchestration:** Configurable agent graphs with short-term runtime execution state and long-term semantic memory.
* **Privacy & Compliance First:** Built-in Role-Based Access Control (RBAC) ensures model requests and RAG search queries strictly respect data boundary permissions.
* **Continuous Self-Improvement Loop:** Failed or high-performing agent runs can be exported directly to `MindSync Training` to continuously fine-tune domain adapters.

---

## 📁 Repository Structure

```tree
mindsync/
├── docs/                     # Architectural specs, API documentation, and guides
├── modules/
│   ├── agents/               # MindSync Agents execution engine and graph workflows
│   ├── knowledge/            # MindSync Knowledge RAG pipelines, chunking, and indexers
│   ├── models/               # MindSync Models abstraction wrappers & inference adapters
│   ├── router/               # MindSync Router rule engine, cost matrix & intent classifiers
│   └── training/             # MindSync Training pipelines, dataset managers & LoRA scripts
├── config/                   # System configurations and security policy templates
├── docker-compose.yml        # Development orchestration
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

* Python >= 3.10
* Docker & Docker Compose
* Node.js >= 18 (if running UI components)

### Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mindsync.git
   cd mindsync
   ```

2. **Environment Configuration:**
   Copy the example environment template and configure your provider keys:
   ```bash
   cp .env.example .env
   ```

3. **Launch Infrastructure:**
   Start vector stores, model caching nodes, and core APIs via Docker Compose:
   ```bash
   docker-compose up -d
   ```

---

## 🛠️ Configuration & Usage

```python
# Example: Basic MindSync Router & Agent invocation
from mindsync.router import RouterEngine
from mindsync.agents import WorkflowAgent

# Initialize Router
router = RouterEngine(config_path="config/router.yaml")

# Route task dynamically
decision = router.route(
    prompt="Summarize the Q3 financial report and flag compliance risks.",
    max_latency_ms=1500,
    cost_sensitivity="balanced"
)

print(f"Selected Agent: {decision.agent_id}")
print(f"Assigned Model: {decision.model_name}")
```

---

## 🛡️ Security & Privacy

MindSync enforces enterprise data boundaries by default:
* Data ingested into **MindSync Knowledge** retains source ACL permissions.
* Prompts passed through **MindSync Models** can be configured to strip PII before external API calls.

---

## 🤝 Contributing

We welcome contributions! Please review our [Contributing Guidelines](docs/CONTRIBUTING.md) prior to submitting Pull Requests.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.