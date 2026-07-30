---
title: DocuLens API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🔍 DocuLens API

AI-powered document Q&A system with hybrid search across PDF, Database, and Chat logs.

## Features
- 📄 PDF Document Analysis
- 🗄️ PostgreSQL Database Query
- 💬 Chat Log Search
- 🔀 Hybrid Search (combine all sources)
- 🤖 Multiple LLM Support (HuggingFace, Ollama, Gemini)

## API Endpoints
- GET /health - Health check
- POST /api/v1/query/hybrid - Hybrid search query
- POST /api/v1/upload - Upload PDF files
- GET /api/v1/collections - List PDF collections
- GET /api/v1/models/available - List available LLM models

## Tech Stack
- FastAPI
- LangChain
- FAISS Vector Store
- PostgreSQL (Neon)
- HuggingFace Transformers

# 📄 PDF QA Assistant / DocuLens (Document Lens)

An intelligent document question-answering system that combines **PDF documents**, **Database**, and **Chat Logs** search with AI-powered responses. Built with FastAPI, LangChain, FAISS, and multiple LLM providers (HuggingFace, Ollama, Gemini).

---

## 🎯 Key Features

### Core Features
- 📄 **PDF Upload & Indexing** - Upload PDFs, automatically chunk and index for semantic search
- 🔍 **Hybrid Search** - Search across PDFs, PostgreSQL database, and chat logs simultaneously
- 🤖 **Multi-LLM Support** - Switch between HuggingFace (local), Ollama (local), or Gemini (cloud)
- 💬 **Chat Log Import** - Import WhatsApp, Telegram, Teams chat exports for searching
- 🎯 **Smart Routing** - Automatically routes queries to relevant data sources

### Advanced Features
- 📊 **Database Integration** - Query structured data with natural language
- 🔗 **PDF Source Links** - Direct links to PDF pages with source text
- ⚡ **Query Expansion** - Automatic synonym and keyword expansion for better recall
- 🌐 **REST API** - Full OpenAPI/Swagger documentation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PDF QA ASSISTANT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   Chat UI   │────▶│              FastAPI Backend                    │   │
│  │  (Next.js)  │◀────│                 (Python)                        │   │
│  └─────────────┘     └─────────────────────────────────────────────────┘   │
│        :3001                           :8000                                │
│                                          │                                  │
│                    ┌─────────────────────┼─────────────────────┐           │
│                    ▼                     ▼                     ▼           │
│           ┌──────────────┐      ┌──────────────┐      ┌──────────────┐    │
│           │  PDF Search  │      │  DB Search   │      │ Chat Search  │    │
│           │   (FAISS)    │      │ (PostgreSQL) │      │   (FAISS)    │    │
│           └──────────────┘      └──────────────┘      └──────────────┘    │
│                    │                     │                     │           │
│                    └─────────────────────┼─────────────────────┘           │
│                                          ▼                                  │
│                              ┌─────────────────────┐                       │
│                              │    LLM Provider     │                       │
│                              │  ┌───────────────┐  │                       │
│                              │  │ HuggingFace   │  │                       │
│                              │  │ Ollama        │  │                       │
│                              │  │ Gemini        │  │                       │
│                              │  └───────────────┘  │                       │
│                              └─────────────────────┘                       │
│                                          │                                  │
│                                          ▼                                  │
│                              ┌─────────────────────┐                       │
│                              │   Generated Answer  │                       │
│                              │   + Source Links    │                       │
│                              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
pdf-reader/
├── main.py                    # FastAPI application entry point
├── config.py                  # Configuration (env vars, models, paths)
├── processor.py               # Core ML pipeline (embeddings, LLM, search)
├── database.py                # PostgreSQL database manager
├── models.py                  # Pydantic request/response models
├── utils.py                   # PDF processing utilities
├── requirements.txt           # Python dependencies
│
├── router/
│   ├── __init__.py
│   ├── upload.py              # PDF upload endpoints
│   ├── collections.py         # PDF collection management + file serving
│   ├── hybrid.py              # Main hybrid search endpoint
│   ├── query.py               # Legacy query endpoint
│   └── chat.py                # Chat log upload & management
│
├── data/
│   ├── uploads/               # Uploaded PDF files (by collection UUID)
│   │   └── {collection-id}/
│   │       └── *.pdf
│   ├── indices/               # FAISS vector indices for PDFs
│   │   └── {collection-id}/
│   │       ├── index.faiss
│   │       └── index.pkl
│   ├── chat_uploads/          # Uploaded chat log files
│   └── chat_indices/          # FAISS vector indices for chats
│
└── .env                       # Environment variables (not in git)
```

---

## 🔄 Application Flow

### Flow 1: PDF Upload & Indexing

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  User    │───▶│ POST /upload│───▶│ PDF Parsing  │───▶│  Chunking   │
│ uploads  │    │   (FastAPI) │    │  (PyPDF2)    │    │ (600 chars) │
│  PDFs    │    └─────────────┘    └──────────────┘    └─────────────┘
└──────────┘                                                  │
                                                              ▼
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│Collection│◀───│ Save Index  │◀───│ FAISS Index  │◀───│ Embeddings  │
│   ID     │    │  to Disk    │    │  Creation    │    │(MiniLM-L12) │
└──────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

**Steps:**
1. User uploads PDF file(s) via `/api/v1/upload`
2. PDFs are saved to `data/uploads/{collection-id}/`
3. Text extracted using PyPDF2
4. Text chunked into 600-char segments with 100-char overlap
5. Chunks embedded using `paraphrase-multilingual-MiniLM-L12-v2`
6. FAISS index created and saved to `data/indices/{collection-id}/`
7. Collection ID returned to user

---

### Flow 2: Hybrid Query (Main Feature)

```
┌──────────┐    ┌─────────────────┐    ┌──────────────────────────────┐
│  User    │───▶│ POST /query/    │───▶│     Question Analysis        │
│  asks    │    │     hybrid      │    │  - Detect keywords           │
│ question │    └─────────────────┘    │  - Expand synonyms           │
└──────────┘                           │  - Route to data sources     │
                                       └──────────────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    ▼                                 ▼                                 ▼
           ┌────────────────┐               ┌────────────────┐               ┌────────────────┐
           │  PDF Search    │               │  DB Search     │               │  Chat Search   │
           │  (FAISS)       │               │  (PostgreSQL)  │               │  (FAISS)       │
           │                │               │                │               │                │
           │ • Similarity   │               │ • Smart table  │               │ • Similarity   │
           │   search       │               │   routing      │               │   search       │
           │ • Top-k docs   │               │ • Full-text    │               │ • Top-k chats  │
           │ • Score > 0.5  │               │   search       │               │ • Score > 0.3  │
           └────────────────┘               └────────────────┘               └────────────────┘
                    │                                 │                                 │
                    └─────────────────────────────────┼─────────────────────────────────┘
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │      Context Preparation     │
                                       │  - Combine all results       │
                                       │  - Truncate to token limit   │
                                       │  - Add source metadata       │
                                       └──────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │        LLM Generation        │
                                       │  - Select provider/model     │
                                       │  - Generate answer           │
                                       │  - Validate output           │
                                       └──────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │         Response             │
                                       │  - Answer text               │
                                       │  - PDF sources with URLs     │
                                       │  - DB results                │
                                       │  - Chat results              │
                                       │  - Processing time           │
                                       └──────────────────────────────┘
```

**Steps:**
1. User sends question to `/api/v1/query/hybrid`
2. System analyzes question:
   - Extracts keywords (e.g., "buyback cash")
   - Expands with synonyms
   - Determines target data sources (PDF/DB/Chat)
3. Parallel search across all sources:
   - **PDF**: FAISS similarity search with score threshold
   - **Database**: Smart table routing + full-text search
   - **Chat**: FAISS similarity search on chat logs
4. Results combined and truncated for LLM context
5. LLM generates answer (HuggingFace/Ollama/Gemini)
6. Response includes answer + source links

---

### Flow 3: View PDF Source

```
┌──────────┐    ┌─────────────────┐    ┌──────────────────────────────┐
│  User    │───▶│ Click "View PDF"│───▶│     PDF Viewer Dialog        │
│  clicks  │    │  in chat UI     │    │  - Opens at specific page    │
│  source  │    └─────────────────┘    │  - Shows source text         │
└──────────┘                           │  - Copy/search functionality │
                                       └──────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │  GET /files/{collection}/    │
                                       │       {filename}#page=N      │
                                       │  - Serves PDF file           │
                                       │  - Browser navigates to page │
                                       └──────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 13+ (optional, for database search)
- Node.js 18+ (for frontend)

### 1. Clone & Setup Backend

```bash
git clone https://github.com/krisnasetyadi/pdf-reader.git
cd pdf-reader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# Database (optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pdf_reader
DB_USER=postgres
DB_PASSWORD=your_password

# CORS
CORS_ORIGINS=http://localhost:3001,http://localhost:3000

# LLM Provider: huggingface | ollama | gemini
LLM_PROVIDER=huggingface
MODEL_NAME=google/flan-t5-base

# Optional: Gemini API (free tier)
# GEMINI_API_KEY=your_api_key

# Optional: Ollama (local)
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2
```

### 3. Run Backend

```bash
uvicorn main:app --reload --port 8000
```

### 4. Setup Frontend (Optional)

```bash
cd ../chat-ui
npm install
npm run dev
```

Open http://localhost:3001

---

## 📡 API Endpoints

### Health Check
```
GET /health
```

### PDF Management
```
POST   /api/v1/upload                    # Upload PDFs
GET    /api/v1/collections               # List all collections
GET    /api/v1/collection/{id}           # Get collection details
DELETE /api/v1/collection/{id}           # Delete collection
GET    /api/v1/files/{collection}/{file} # Serve PDF file
```

### Hybrid Search (Main)
```
POST   /api/v1/query/hybrid              # Search PDFs + DB + Chats
GET    /api/v1/models/available          # List available LLM models
```

### Chat Logs
```
POST   /api/v1/chat/upload               # Upload chat export
GET    /api/v1/chat/collections          # List chat collections
DELETE /api/v1/chat/collection/{id}      # Delete chat collection
```

---

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `huggingface` | LLM provider (huggingface/ollama/gemini) |
| `MODEL_NAME` | `google/flan-t5-base` | HuggingFace model |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence embeddings |
| `CHUNK_SIZE` | `600` | Text chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `K_RESULTS` | `5` | Results per search |
| `TEMPERATURE` | `0.3` | LLM temperature |

---

## 🌐 Deployment

### Free Deployment Stack
- **Frontend**: Vercel (unlimited)
- **Backend**: Hugging Face Spaces (16GB RAM free)
- **Database**: Neon PostgreSQL (512MB free)

See deployment guide in `/docs/DEPLOYMENT.md`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| LLM | HuggingFace Transformers / Ollama / Gemini |
| Embeddings | sentence-transformers (MiniLM) |
| Vector Store | FAISS |
| Orchestration | LangChain |
| Database | PostgreSQL |
| Frontend | Next.js 16 + React 19 |
| UI Components | shadcn/ui + Radix UI |
| Styling | Tailwind CSS |

---

## 📄 License

MIT License - see LICENSE file

---

## 👤 Author

**Krisna Setyadi**
- GitHub: [@krisnasetyadi](https://github.com/krisnasetyadi)