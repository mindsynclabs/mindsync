# MindSync RAG Prototype

A small, fully local Retrieval-Augmented Generation (RAG) service. Upload PDFs or insert text, then chat with your documents. Everything runs on your own machine: no cloud APIs required.

**Stack:** FastAPI · LlamaIndex · LanceDB · Ollama · sentence-transformers

> ### Runs on a plain CPU, no GPU needed
> This whole stack runs on a **CPU-only machine**, for example **4 CPU cores and 32 GB of RAM**. Chat, embeddings, search and PDF upload all work without a GPU, using small local models such as `qwen2.5:0.5b`. A GPU is optional and only makes it faster (see [Hardware: CPU or GPU](#hardware-cpu-or-gpu)).
>
> **What to expect on CPU:** text ingestion and chat are usable with small models, but replies take longer than on a GPU. Scanned-PDF OCR with a 7B vision model is the slowest part (minutes per page), so keep `OCR_MAX_PAGES` low. Text-based PDFs are fast either way.

> **Lab scope:** this repo is a learning prototype that implements only a few functions: insert text, upload a file (with OCR for scanned PDFs), and chat. It is **not** a full product. The design is deliberately small and modular, so it can grow to cover many more data sources (email, Google Drive, and so on) and decentralized storage such as Filecoin. See [Extending the project](#extending-the-project) for how, and for what is implemented versus only designed.

## Features

- **Insert text** or **upload files** (`.pdf`, `.txt`, `.md`)
- **Scanned PDF support:** pages without extractable text are read by a vision model (`qwen2.5vl:7b`) through Ollama
- **Chat with memory:** each `session_id` keeps its own conversation history
- **Source citations:** every answer returns the chunks it was based on
- **Re-ingest safe:** uploading the same `doc_id` replaces the old chunks instead of duplicating them
- **Local-first:** embeddings, vector store and LLM all run locally
- **Runs on CPU or GPU:** works on a plain CPU-only machine, and speeds up automatically when a GPU is available (see [Hardware: CPU or GPU](#hardware-cpu-or-gpu))

## How it works

```
                 ┌───────────────┐
 text / file ──► │ FastAPI       │
                 └──────┬────────┘
                        │ PDF? text layer → pypdf
                        │      no text    → render page → qwen2.5vl (OCR)
                        ▼
              chunk (SentenceSplitter, 256 tokens, 32 overlap)
                        ▼
              embed (all-MiniLM-L6-v2)  ──►  LanceDB

 question ──► embed ──► top-k chunks from LanceDB ──► qwen2.5 (Ollama) ──► answer + sources
```

## Requirements

- **No GPU required.** A CPU-only machine works; a GPU just makes it faster
- Python 3.12 or 3.13 recommended (very new versions may lack wheels for some ML packages)
- [Ollama](https://ollama.com) installed and running (a recent version is needed for `qwen2.5vl`)
- Internet access once, to download the embedding model on first start

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (always use `python -m pip` so it targets the venv)
python -m pip install -r requirements.txt

# 3. Pull the models
ollama pull qwen2.5:0.5b      # chat model
ollama pull qwen2.5vl:7b      # OCR model (only needed for scanned PDFs)
```

## Run

From the project root (the folder that contains `src/`):

```bash
python -m uvicorn main:app --app-dir src --reload
```

Wait for `Application startup complete.` (the first start loads the embedding model and can take a while), then open **http://localhost:8000/docs** for the interactive API docs.

## Hardware: CPU or GPU

The project runs on **either**. It was built and tested on the assumption of a CPU-only machine, and every part also benefits from a GPU when one is present. No code changes are needed to switch.

| Component | On CPU | On GPU |
|---|---|---|
| **Chat LLM** (Ollama) | Works. Fine for small models (0.5B to 3B), slower for larger ones | Ollama uses a supported GPU automatically (NVIDIA CUDA, AMD ROCm, Apple Silicon Metal). Much faster, and larger models become practical |
| **OCR vision model** (Ollama) | Works but slow: minutes per page for a 7B model | Typically many times faster; the main reason to get a GPU for scanned PDFs |
| **Embeddings** (sentence-transformers) | Works well. `all-MiniLM-L6-v2` is small and fast even on CPU | Auto-detected if PyTorch can see the GPU. Helps most when ingesting large volumes |
| **Vector search** (LanceDB) | CPU, fast for prototype-sized data | Same (no GPU needed) |
| **PDF page rendering** (PyMuPDF) | CPU | CPU |

### Suggested model choices

| Hardware | Chat model | OCR model |
|---|---|---|
| CPU only | `qwen2.5:0.5b` to `qwen2.5:3b` | `qwen2.5vl:3b` or `qwen2.5vl:7b` with a low `OCR_MAX_PAGES` |
| GPU with modest VRAM | `qwen2.5:3b` to `qwen2.5:7b` | `qwen2.5vl:7b` |
| GPU with plenty of VRAM | a larger `qwen2.5` model | `qwen2.5vl:7b` or larger |

These are starting points, not benchmarks. If a model does not fit in GPU memory, Ollama splits it between GPU and CPU, which is slower than a model that fits entirely.

### Check what is actually being used

```bash
ollama ps          # the PROCESSOR column shows 100% GPU, 100% CPU, or a split
nvidia-smi         # NVIDIA only: GPU memory use while a request runs
python -c "import torch; print(torch.cuda.is_available())"   # can PyTorch (embeddings) see the GPU?
```

### Forcing a device for embeddings

By default the embedding model picks a GPU if PyTorch finds one, otherwise CPU. To force it:

```bash
EMBED_DEVICE=cpu  python -m uvicorn main:app --app-dir src --reload   # keep the GPU free for Ollama
EMBED_DEVICE=cuda python -m uvicorn main:app --app-dir src --reload   # NVIDIA GPU
```

On a machine where the GPU is busy running a large chat or OCR model, forcing embeddings to `cpu` is often a good choice, since the embedding model is small.

## Project structure

```
mindsync/
├── src/
│   ├── main.py        # FastAPI app and endpoints
│   ├── rag.py         # LlamaIndex + LanceDB + Ollama: ingest and chat
│   ├── ocr.py         # PDF text extraction with vision-model OCR fallback
│   └── telegram_service.py  # optional Telegram bot inside the FastAPI app
├── requirements.txt
├── lancedb_data/      # created on first ingest (the vector database)
└── README.md
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Shows the configured LLM and embedding model |
| `POST` | `/documents/text` | Insert text |
| `POST` | `/documents/upload` | Upload a `.pdf`, `.txt` or `.md` file |
| `DELETE` | `/documents/{doc_id}` | Remove a document's chunks |
| `POST` | `/chat` | Ask a question |
| `DELETE` | `/chat/{session_id}` | Clear a conversation's memory |

### Examples

**Insert text**
```bash
curl -X POST http://localhost:8000/documents/text \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "notes", "text": "Aurora Labs was founded in 2021 in Da Nang.", "metadata": {"source": "notes"}}'
```

**Upload a PDF**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@handbook.pdf" \
  -F "doc_id=handbook"
```

**Chat**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "When was Aurora Labs founded?", "session_id": "s1"}'
```

Response:
```json
{
  "answer": "Aurora Labs was founded in 2021.",
  "sources": [
    {"doc_id": "notes", "score": 0.82, "snippet": "Aurora Labs was founded in 2021 ...", "metadata": {}}
  ]
}
```

> Chatting before any document is ingested returns `409`. Add a document first.

## Configuration

All settings are environment variables with sensible defaults.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Chat model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EMBED_DEVICE` | auto | `cpu`, `cuda` or `mps` to force the embedding device; empty = auto-detect |
| `DB_PATH` | `./lancedb_data` | LanceDB folder |
| `TABLE_NAME` | `rag_documents` | LanceDB table |
| `TOP_K` | `3` | Chunks retrieved per question |
| `CHAT_MODE` | `condense_plus_context` | `context` = one LLM call per message (faster); `condense_plus_context` = rewrites follow-up questions first |
| `PDF_OCR` | `auto` | `auto` = OCR only pages with no text; `always`; `off` |
| `OCR_MODEL` | `qwen2.5vl:7b` | Vision model used for OCR |
| `OCR_MAX_PAGES` | `10` | Max pages OCRed per upload |
| `OCR_DPI` | `150` | Render resolution for OCR (lower = faster) |
| `OCR_MIN_CHARS` | `30` | A page with fewer extracted characters is treated as scanned |

Example:
```bash
OLLAMA_MODEL=qwen2.5:1.5b CHAT_MODE=context OCR_DPI=100 \
  python -m uvicorn main:app --app-dir src --reload
```

## Telegram (optional)

You can chat with your documents from Telegram. The bot lives **inside the FastAPI app** (`src/telegram_service.py`): one process, and it calls the same RAG functions as the HTTP endpoints. It stays off unless you set a bot token.

**What you can do in Telegram**
- Send a text message to ask a question (each Telegram chat has its own conversation memory)
- Send a PDF, `.txt` or `.md` file to add it (a caption becomes the `doc_id`)
- `/reset` clears the conversation memory, `/delete <doc_id>` removes a document, `/help` lists the commands

**Setup**
1. In Telegram, talk to **@BotFather**, send `/newbot`, and copy the token.
2. Start the app with the token set. The bot is locked until you allow your user id, and it will tell you your id when you message it:
   ```bash
   export TELEGRAM_BOT_TOKEN="123456:ABC..."
   python -m uvicorn main:app --app-dir src
   ```
3. Message the bot, copy the user id it replies with, then restart with it allowed:
   ```bash
   export TELEGRAM_ALLOWED_IDS="11111111"      # comma-separated for several people
   ```

**Two modes (`TELEGRAM_MODE`)**

| Mode | How it works | Needs |
|---|---|---|
| `polling` (default) | The app asks Telegram for new messages in the background | Nothing public. Works behind a home router. Use a single uvicorn worker |
| `webhook` | Telegram calls `POST /telegram/webhook` on your FastAPI app | A public HTTPS URL (a server, or a tunnel such as Cloudflare Tunnel or ngrok) |

Webhook mode also needs:
```bash
export TELEGRAM_MODE=webhook
export TELEGRAM_WEBHOOK_URL="https://your-public-host"
export TELEGRAM_WEBHOOK_SECRET="a-long-random-string"   # checked on every call
```

`GET /telegram/status` shows whether the bot is running and in which mode.

**Security**
- Keep the token secret and out of git. If it leaks, revoke it with `/revoke` in @BotFather.
- Only user ids in `TELEGRAM_ALLOWED_IDS` get answers. Everyone else is ignored, and groups and channels are ignored too.
- Messages and files pass through Telegram's servers. The LLM, embeddings and database stay local.

**Limits:** replies are split at 4000 characters, and Telegram bots can download files up to 20 MB. Scanned PDFs can take minutes on CPU, so the bot shows "typing..." while it works.

`telegram_bot.py` is an alternative: a standalone script that talks to the API over HTTP, for running the bot as a separate process or on another machine. Use one or the other, not both with the same token.

## Extending the project

The core of the service is one function, `rag.ingest_text(doc_id, text, metadata)`. Anything that can produce text plus an ID can feed it, so new sources are mostly a matter of writing a small "adapter" in front of it. The query side does not change.

```
 Sources (connectors)                 Core pipeline                    Storage
 ─────────────────────               ───────────────────────────      ───────────────
 text API         [done]  ┐
 file upload      [done]  │
 PDF + OCR        [done]  ├──► Documents ──► normalize metadata ──►    LanceDB (local) [done]
 Google Drive     [idea]  │    (text + id)    chunk → embed             raw files + index
 Gmail / email    [idea]  │                                             snapshots on
 Notion, Slack... [idea]  │                                             Filecoin/IPFS [idea]
 folder watcher   [idea]  ┘
```

### 1. Many data sources (LlamaIndex connectors)

LlamaIndex has a connector system called **LlamaHub**. A connector (a "Reader") loads data from a source and returns LlamaIndex `Document` objects, which is exactly what this project's pipeline consumes. LlamaHub lists hundreds of readers, including local folders (`SimpleDirectoryReader`, built in), Notion, Google Docs, Slack, Discord, web pages, databases, Obsidian and mbox email files. Browse https://llamahub.ai/ for the current list and the exact package names (for example, Google Drive and Gmail readers live in the Google readers package).

Because every reader produces the same `Document` type, you can mix sources in one index.

Sketch of a connector adapter (not included in this lab):

```python
# pip install llama-index-readers-google   (needs Google OAuth credentials)
from llama_index.readers.google import GoogleDriveReader
import rag

def sync_drive_folder(folder_id: str):
    docs = GoogleDriveReader().load_data(folder_id=folder_id)
    for d in docs:
        rag.ingest_text(
            doc_id=d.doc_id,
            text=d.text,
            metadata={"source": "gdrive", **d.metadata},  # extra keys are kept in "extra"
        )
```

Then expose it as an endpoint (for example `POST /connectors/gdrive/sync`) or run it on a schedule. Email works the same way: a reader yields one document per message, and each message becomes a `doc_id`. Re-syncing is safe because `ingest_text` replaces the old chunks for a `doc_id`.

Things to plan for when adding connectors:
- **Authentication:** each source needs its own credentials (OAuth, API tokens). Keep them in environment variables or a secrets manager, never in the repo.
- **Incremental sync:** store a content hash per document (`file_hash` already exists) and skip unchanged items.
- **Permissions and privacy:** email and drive content is sensitive, so decide who can query it. This prototype has no authentication on its endpoints.
- **Metadata schema:** LanceDB fixes its columns on the first insert, so new sources should reuse the fixed keys (`source`, `filename`, `file_hash`) and put everything else in `extra`.

### 2. Decentralized file storage (Filecoin / IPFS)

The idea is to keep a **local LanceDB index for fast queries** and use a decentralized network as the **durable, verifiable store** for the raw files and for index snapshots. Queries never touch the network, so they stay fast.

```
 upload ──► save raw file ──► upload to Filecoin/IPFS ──► get CID ──► store CID in metadata
                │
                └──► parse → chunk → embed → LanceDB (local, hot path)

 every N minutes: zip lancedb_data ──► upload ──► new manifest (CIDs of files + snapshot + model name)
 restore: manifest CID ──► download snapshot and files ──► rebuild a working index
```

What it would add:
- **Provenance:** each answer can cite the CID of the exact file it came from, and a verify step can re-fetch the CID and check its hash.
- **Tamper evidence and durability:** content-addressed data changes its CID if it is edited, and Filecoin's storage proofs show the data is still being held.
- **Restore from a pointer:** wipe the machine and rebuild the index from a single manifest CID.

How it would be built:
- Add a `file_cid` key to the fixed metadata keys (then reset the table, since the schema changes).
- Write a small storage module with `put(bytes) -> cid` and `get(cid) -> bytes`. The official Filecoin Onchain Cloud SDK (Synapse SDK) is TypeScript, so from this Python service you would call a small Node sidecar or CLI, or use an IPFS HTTP gateway or pinning service instead.
- Upload in the background through a queue, so a slow or failed upload never blocks ingestion or chat.
- Record the embedding model name and chunking settings in the manifest, so a restored index stays consistent.
- Files on public networks are visible to others by default. **Encrypt sensitive files before uploading** and keep the keys local.

What it does not do: storage proofs show that data is stored and unchanged, not that an answer is correct.

### 3. Other directions

- **Folder-watcher daemon:** watch `./knowledge/` and call `ingest_text` automatically for new or changed files.
- **Background jobs:** return a job ID from upload and process in the background, so long OCR runs don't hit client timeouts.
- **Better answers:** a larger chat model, a reranker, or hybrid (keyword + vector) search.
- **Fine-tuning:** train a small LoRA adapter for style and format, keep facts in RAG, and store the adapter and dataset alongside the index.

## Notes and limitations

- **Small models are weak.** `qwen2.5:0.5b` is fast but often ignores "answer only from context" and may hallucinate. Try `qwen2.5:1.5b` or `3b` for better answers.
- **OCR is slow on CPU, much faster on GPU.** A 7B vision model takes minutes per page on CPU. On CPU, lower `OCR_DPI` and `OCR_MAX_PAGES`, or try `qwen2.5vl:3b`. With a GPU (check with `ollama ps`) it is far quicker. Text-based PDFs are not affected either way.
- **Metadata schema is fixed.** LanceDB fixes its columns on the first insert, so every document is stored with the same keys: `source`, `filename`, `file_hash`, and `extra` (any other keys, as JSON).
- **Chat sessions live in memory** and reset when the server restarts.
- **The upload request waits for ingestion** (including OCR) to finish.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn: command not found` | Use `python -m uvicorn ...` inside the activated venv |
| `No module named 'xyz'` | Run `python -m pip install -r requirements.txt` in the activated venv |
| `Table rag_documents is not initialized` | No documents yet. Ingest one first |
| LanceDB error when adding documents | Schema mismatch from an older run. Stop the server, `rm -rf lancedb_data`, and re-ingest |
| `Chat failed (is Ollama running ...)` | Start Ollama and check the model is pulled: `ollama list` |
| OCR error about the model | `ollama pull qwen2.5vl:7b`, and update Ollama if the pull is refused |
| Scanned PDF returns "No text found" | Check `PDF_OCR` is not `off`, and that the OCR model is pulled |

## Fine-tune
* **Hardware Notes:** 

  * **NVIDIA GPU (CUDA):** Automatically uses bfloat16 (Ampere/Ada/Hopper) or float16.
  * **Apple Silicon (MPS):** Supports float16 on M1/M2/M3/M4 chips.
  * **CPU:** Supported out-of-the-box in float32 (slower execution).

#### 1. Dataset Format (ChatML)

Training data is stored in sample_data/fine_tune_data_set.jsonl using Qwen 2.5's native multi-turn messages format: 

json

{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant for MindSync Labs."},
    {"role": "user", "content": "Who founded MindSync Labs and when?"},
    {"role": "assistant", "content": "MindSync Labs was founded in 2024 by Thang Tran in Ho Chi Minh City, Vietnam."}
  ]
}

#### Dataset Design Principles

* **Domain Coverage (35 samples):** Covers direct questions, paraphrased queries, negative constraints, and multi-turn conversations about MindSync Labs and MindSync.
* **Replay Data (15 samples):** General coding, math, and factual QA pairs (~30% of the dataset) prevent catastrophic forgetting and stop the model from overfitting on company keywords.
* **Loss Masking:** System and user tokens are masked (-100) during training so loss is computed strictly on assistant responses.

#### 2. Fine-Tuning

Run the training script with unbuffered output (-u) to stream real-time step and epoch metrics: 

bash

python -u src/fine_tune.py \
  --model-id "Qwen/Qwen2.5-3B-Instruct" \
  --data-path ./sample_data/fine_tune_data_set.jsonl \
  --output-dir ./lora_adapter \
  --epochs 10 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --lr 2e-4 \
  --val-ratio 0.0

*For GPUs with 8GB VRAM, reduce memory usage with --batch-size 1 --grad-accum-steps 4.* 

#### Training Configuration Summary

* **LoRA Rank / Alpha:** r=16, lora_alpha=32, lora_dropout=0.05
* **Target Modules:** All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
* **Scheduler:** Cosine learning rate schedule with 10% warmup (warmup_ratio=0.1)
* **Artifacts:** Saves the best adapter weights and training_metrics.json inside ./lora_adapter/.

#### 3. Evaluation

Verify that the fine-tuned adapter accurately recalls MindSync Labs facts without hallucinating entities or degrading general coding performance: 

bash

python test_model.py


#### 4. Merging LoRA Weights

Merge the trained LoRA adapter into the base Qwen2.5-3B-Instruct model to create a standalone directory (./mindsync-qwen2.5-3b) with no peft runtime dependency: 

bash

python merge_model.py

Use code with caution.

Copy code text to clipboard.

#### 5. Local RAG Deployment

#### Option A: Deploy with Ollama

1. Create the local model from the merged Safetensors directory using the included Modelfile (which configures Qwen 2.5's <|im_start|> / <|im_end|> stop tokens):

bash

ollama create mindsync-3b -f Modelfile

Use code with caution.

Copy code text to clipboard.

1. Query the model in your RAG pipeline via the local Ollama API:

python

import ollama

def query_rag(question: str, context: str) -> str:
    response = ollama.chat(
        model="mindsync-3b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant for MindSync Labs. Use the provided context to answer the user's question accurately. If the answer is not in the context, state that clearly.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response["message"]["content"]


#### Option B: Deploy with Hugging Face Transformers

Load ./mindsync-qwen2.5-3b directly in Python: 

python

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./mindsync-qwen2.5-3b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()


## Roadmap

Implemented in this lab: text insert, file upload, PDF OCR fallback, chat with memory and sources, document delete.

Not implemented yet (see [Extending the project](#extending-the-project)):

- [x] Telegram bot (optional, inside the FastAPI app; untested in this lab)
- [ ] Streaming chat endpoint (`/chat/stream`)
- [ ] Background ingestion jobs with status polling
- [ ] Faster OCR path (Tesseract first, vision model as fallback)
- [ ] Folder-watcher daemon for automatic ingestion
- [ ] LlamaIndex connectors: Google Drive, Gmail/email, Notion, Slack, web pages
- [ ] Filecoin/IPFS storage for raw files and index snapshots, with CIDs in citations
- [ ] Authentication on the API