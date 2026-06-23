<<<<<<< HEAD
# fmai
fmai is an offline AI system that can understand and retrieve information from multiple data formats such as documents (PDF/DOC), images, and voice recordings. It converts all inputs into a unified semantic representation using embeddings and stores them in a local database. 
=======
# 🔥 MistralRAG — Multi-Format Document Intelligence

A powerful RAG (Retrieval-Augmented Generation) system powered entirely by **Mistral AI**. Upload PDFs, documents, and audio files, then chat with your knowledge base through a premium web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Mistral AI](https://img.shields.io/badge/Mistral_AI-Powered-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=flat-square)

## ✨ Features

- **Multi-Format Ingestion**: PDFs, DOCX, TXT, and audio files (MP3, WAV, M4A, FLAC, OGG)
- **100% Mistral Ecosystem**: Embeddings (`mistral-embed`), transcription (`voxtral-mini-latest`), and generation (`mistral-large-latest`)
- **Smart Chunking**: Overlapping text chunks preserve context at boundaries
- **Source Attribution**: Every answer includes references to source documents
- **Premium Web UI**: Dark-themed glassmorphism design with drag & drop upload
- **Persistent Storage**: ChromaDB stores embeddings to disk — your knowledge base survives restarts
- **Document Management**: Upload, view, and delete documents through the UI

## 🏗️ Architecture

```
User Query → Embed Query (mistral-embed)
           → Retrieve Top-K Chunks (ChromaDB)
           → Generate Answer (mistral-large-latest)
           → Return Answer + Source Citations
```

## 📋 Prerequisites

- **Python 3.10+**
- **Mistral AI API Key** — Get one at [console.mistral.ai](https://console.mistral.ai)

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
cd rag
```

### 2. Set Up Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure API Key
```bash
# Copy the example env file
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux

# Edit .env and add your Mistral API key
# MISTRAL_API_KEY=your_actual_api_key_here
```

### 5. Run the Application
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the UI
Navigate to **http://localhost:8000** in your browser.

## 📁 Supported File Formats

| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| PDF | `.pdf` | PyMuPDF (fitz) — text extraction with page numbers |
| Word | `.docx` | python-docx — paragraphs, tables, headings |
| Text | `.txt` | Direct file reading with encoding detection |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` | Voxtral (Mistral API) — speech-to-text |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and process a document |
| `POST` | `/api/chat` | Send a query and get RAG response |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `GET` | `/api/health` | Health check |

## 🧠 How It Works

1. **Upload**: Drop a file into the web UI
2. **Process**: The system extracts text (or transcribes audio), splits it into overlapping chunks
3. **Embed**: Each chunk is embedded using `mistral-embed` (1024-dim vectors)
4. **Store**: Embeddings are stored in ChromaDB with metadata
5. **Query**: When you ask a question, it's embedded and matched against stored chunks
6. **Generate**: The top-K relevant chunks are sent as context to `mistral-large-latest`
7. **Respond**: You get an answer with citations pointing to source documents

## 📂 Project Structure

```
rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app & routes
│   │   ├── config.py            # Settings
│   │   ├── models.py            # Pydantic schemas
│   │   ├── document_processor.py # File parsers
│   │   ├── chunker.py           # Text splitter
│   │   ├── embeddings.py        # Mistral embeddings
│   │   ├── vector_store.py      # ChromaDB ops
│   │   └── rag_engine.py        # RAG pipeline
│   ├── requirements.txt
│   ├── .env.example
│   ├── uploads/                 # Uploaded files
│   └── chroma_db/               # Vector database
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
└── README.md
```

## ⚠️ Notes

- Audio files up to ~100MB are supported
- The Mistral API key is stored in `.env` (never commit this file!)
- ChromaDB data persists in `backend/chroma_db/`
- Uploaded files are stored in `backend/uploads/`

## 📄 License

MIT License — feel free to use and modify.
>>>>>>> master
