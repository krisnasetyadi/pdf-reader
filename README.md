# 📄 PDF QA API

A FastAPI-based API for uploading PDF documents, indexing them, and querying their content using natural language. It uses HuggingFace Transformers, FAISS for vector search, and LangChain for question answering.

---

## 🚀 Features

- Upload PDF files and create searchable collections
- Query documents using natural language
- Get answers with source document references
- View and manage uploaded document collections
- Uses pretrained models (`google/flan-t5-large`, `all-MiniLM-L6-v2`)
- Powered by HuggingFace, FAISS, and LangChain

---

## 📦 Project Structure

```
app/
├── main.py # FastAPI app and router setup
├── config/ # Configuration settings
├── processor/ # ML pipeline setup
├── router/
│ ├── upload.py # Upload PDF files
│ ├── query.py # Query PDFs with natural language
│ └── collections.py # Manage document collections
├── models.py # Pydantic models
├── utils.py # PDF processing utilities
```

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run The Server

```bash
uvicorn app.main:app --reload
```

### Configuration

The default configuration is defined in config/config.py.
You can adjust:

    model_name: "google/flan-t5-large"
    embedding_model: "all-MiniLM-L6-v2"

    For document chunking:

    chunk_size:1000
    chunk_overlap: 200

    Storage paths:
    upload_folder: "uploads"
    index_folder: "indices"

# API Documentation

## 🔼 Upload PDFs

`POST /upload`

Upload one or more PDF files to create a new collection.

### Request (multipart/form-data)

- `files`: List of PDF files

### Response (200 OK)

```json
{
  "collection_id": "string",
  "file_count": 3,
  "status": "success"
}
```

```bash
{
  "answer": "The main topic is...",
  "sources": [
    "document1.pdf (page 2)",
    "document2.pdf (page 5)"
  ],
  "collection_id": "your-collection-id",
  "processing_time": 1.23
}
```

```json
{
  "answer": "The main topic is...",
  "sources": ["document1.pdf (page 2)", "document2.pdf (page 5)"],
  "collection_id": "your-collection-id",
  "processing_time": 1.23
}
```

## ❓ Query Documents

`POST /query`

Query the content of your uploaded PDFs using natural language.

### Request Body (JSON)

```json
{
  "question": "What is the main topic of the document?",
  "collection_id": "optional-collection-id",
  "include_sources": true
}
```

```json
{
  "answer": "The main topic is...",
  "sources": ["document1.pdf (page 2)", "document2.pdf (page 5)"],
  "collection_id": "your-collection-id",
  "processing_time": 1.23
}
```

### Collections

`GET /collections`

Get a list of available document collections.

Response (200 OK)

```json
[
  {
    "collection_id": "abc123",
    "document_count": 2,
    "created_at": "2025-05-20T10:00:00",
    "file_names": ["report1.pdf", "report2.pdf"]
  }
]
```

`DELETE /collection/{collection_id}`

Delete a specific document collection by its ID.

Path Parameter

- collection_id: The ID of the collection to delete

```json
{
  "message": "Collection deleted successfully"
}
```
