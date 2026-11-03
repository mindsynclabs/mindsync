#!/usr/bin/env bash

# Thoát ngay lập tức nếu có lỗi xảy ra
set -e

PROJECT_NAME="mindsync"

echo "🚀 Đang khởi tạo cấu trúc thư mục cho dự án: $PROJECT_NAME..."

# 1. Tạo các thư mục gốc và dữ liệu cục bộ
mkdir -p data/lancedb
mkdir -p data/uploads
mkdir -p tests

# 2. Tạo cấu trúc thư mục trong src/
mkdir -p src/core
mkdir -p src/modules/knowledge
mkdir -p src/modules/models
mkdir -p src/modules/agents/tools
mkdir -p src/modules/agents/core_agents
mkdir -p src/modules/training
mkdir -p src/api/endpoints

# 3. Tạo các file __init__.py để Python nhận diện module
touch src/__init__.py
touch src/core/__init__.py
touch src/modules/__init__.py
touch src/modules/knowledge/__init__.py
touch src/modules/models/__init__.py
touch src/modules/agents/__init__.py
touch src/modules/agents/tools/__init__.py
touch src/modules/agents/core_agents/__init__.py
touch src/modules/training/__init__.py
touch src/api/__init__.py
touch src/api/endpoints/__init__.py
touch tests/__init__.py

# 4. Tạo các file mã nguồn nòng cốt
touch src/config.py
touch src/core/security.py
touch src/core/telemetry.py

touch src/modules/knowledge/ingest.py
touch src/modules/knowledge/store.py
touch src/modules/knowledge/retriever.py

touch src/modules/models/provider.py
touch src/modules/models/embeddings.py

touch src/modules/agents/deps.py
touch src/modules/agents/schemas.py
touch src/modules/agents/tools/rag_tool.py
touch src/modules/agents/core_agents/knowledge_agent.py
touch src/modules/agents/core_agents/router_agent.py

touch src/modules/training/dataset.py
touch src/modules/training/trainer.py

touch src/api/router.py
touch src/api/endpoints/chat.py
touch src/api/endpoints/knowledge.py
touch src/api/endpoints/health.py

touch tests/test_ingest.py
touch tests/test_agents.py

# 5. Tạo file .env.example
cat << 'EOF' > .env.example
# MindSync Environment Configurations
APP_NAME=MindSync
DEBUG=True

# LLM & Embedding Settings (Ollama / Local)
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=qwen2.5:7b

# Data Paths
LANCEDB_PATH=./data/lancedb
UPLOAD_DIR=./data/uploads
EOF

# 6. Tạo file .gitignore
cat << 'EOF' > .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
env/
venv/

# Environment Variables
.env

# Data & Database Local Storage (Privacy-First)
data/lancedb/*
data/uploads/*
!data/lancedb/.gitkeep
!data/uploads/.gitkeep

# IDEs
.vscode/
.idea/
EOF

# Giữ lại thư mục rỗng data trong git nhưng bỏ qua nội dung bên trong
touch data/lancedb/.gitkeep
touch data/uploads/.gitkeep

# 7. Tạo file pyproject.toml mẫu (Dùng cho Poetry / UV)
cat << 'EOF' > pyproject.toml
[tool.poetry]
name = "mindsync"
version = "0.1.0"
description = "Enterprise Open-Source AI Platform (Local-First)"
authors = ["MindSync Developer"]

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.110.0"
uvicorn = "^0.28.0"
pydantic = "^2.6.0"
pydantic-ai = "^0.0.1"
lancedb = "^0.6.0"
httpx = "^0.27.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF

# 8. Tạo file src/main.py cơ bản
cat << 'EOF' > src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="MindSync Engine",
    description="Enterprise Open-Source AI Platform (Local-First)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "MindSync Engine is running smoothly!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
EOF

# 9. Tạo README.md
cat << 'EOF' > README.md
# MindSync Engine

Enterprise Open-Source AI Platform with Local-First & Privacy-First Architecture.

## Tech Stack
- **API Gateway:** FastAPI
- **Agent Engine:** PydanticAI
- **Knowledge Base:** LanceDB
- **Inference:** Ollama / vLLM
- **Fine-Tuning:** Unsloth

## How to Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # hoặc poetry install
python src/main.py