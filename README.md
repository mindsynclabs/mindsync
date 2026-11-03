Markdown
# MindSync Engine 🧠⚡

Enterprise Open-Source AI Platform with a **Local-First**, **Privacy-First** Architecture. Designed for modularity, absolute data sovereignty, and zero-telemetry operations.

---

## 🏛️ Architecture Overview

MindSync follows a clean **Modular Monolith** pattern within a structured `src/` package, separating core domain engines from the FastAPI gateway layer:

```text
mindsync/
├── data/                       # Local data storage (Ignored by Git)
│   ├── lancedb/                # Embedded Vector Database
│   └── uploads/                # Raw documents (PDF, MD, TXT)
├── src/                        # Main source code package
│   ├── api/                    # FastAPI endpoints & routers
│   ├── core/                   # Shared configurations & security
│   └── modules/                # Core domain engines
│       ├── knowledge/          # RAG pipeline, chunking & LanceDB
│       ├── models/             # Local LLM / Embedding providers (Ollama/vLLM)
│       ├── agents/             # PydanticAI agents & tools
│       └── training/           # Unsloth LoRA fine-tuning pipeline
├── tests/                      # Unit & integration tests
├── pyproject.toml              # Project dependencies & metadata
└── requirements.txt            # Locked pip dependencies
🚀 Quick Start (Local Setup)
1. Clone & Navigate
Bash
git clone [https://github.com/your-username/mindsync.git](https://github.com/your-username/mindsync.git)
cd mindsync
2. Setup Virtual Environment
Bash
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the FastAPI Development Server
Bash
python3 -m uvicorn src.main:app --reload --port 8000
API Endpoint: http://localhost:8000/

Interactive API Docs (Swagger UI): http://localhost:8000/docs

🛠️ Tech Stack
API Gateway: FastAPI (Stateless, high-performance async REST)

Agent Engine: PydanticAI (Type-safe, structured LLM agents)

Knowledge Base: LanceDB (Serverless, embedded local vector database)

Inference Engine: Ollama / vLLM (Running local models like qwen2.5:7b / llama3.2:3b)

Fine-Tuning: Unsloth (Efficient LoRA optimization pipeline)

🔒 Privacy & Security Controls
No Telemetry: Zero third-party data leaks, tracking analytics, or cloud logging.

Stateless Design: No chat history is permanently persisted on server memory or external databases unless explicitly managed locally.

📜 License
MIT License