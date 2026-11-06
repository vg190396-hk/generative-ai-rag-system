# Generative AI–Powered Retrieval-Augmented Generation (RAG) System

A document question-answering application built with Python, LangChain, OpenAI Embeddings, FAISS, Streamlit and PyPDF.

Upload PDF or TXT documents, ask questions in natural language, and receive answers based on retrieved passages from those documents. The interface displays the source filename, page number when available, and a supporting excerpt.

## Features

- Upload multiple PDF and TXT files, or provide article URLs.
- Extract PDF text page by page using PyPDF.
- Fetch and preprocess article content from URLs using LangChain `UnstructuredURLLoader`.
- Preserve source filename and page metadata.
- Split long text into overlapping chunks using LangChain.
- Generate vector embeddings using OpenAI Embeddings.
- Store and search vectors using FAISS.
- Retrieve relevant passages for a user question.
- Generate context-grounded answers using ChatOpenAI.
- Display source filenames, page references where available, and excerpts.
- Keep API credentials in a local `.env` file.

## Architecture

`Upload PDF/TXT or fetch URL → Extract → Chunk → Embed → FAISS → Retrieve → ChatGPT → Answer with sources`

## Requirements

- Python 3.10 or 3.11 recommended.
- An OpenAI API key with API access and available billing/credits.
- Internet connection for OpenAI API calls and package installation.

## Setup (Windows / VS Code)

1. Open this folder in VS Code.
2. Open the integrated terminal.
3. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Open `.env` and replace `your_openai_api_key_here` with your own API key.

6. Start the app:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

The application code and repository have been uploaded. Full live testing of the OpenAI-powered workflow requires an API key with available credits.

## How to use

1. Open the app in your browser.
2. Upload one or more `.pdf` or `.txt` files in the sidebar.
3. Optionally paste one article URL per line in the sidebar. Click **Process documents** and wait for indexing to finish.
4. Enter a question about the uploaded material.
5. Click **Generate answer**.
6. Review the answer and expand **Sources used** to inspect filenames, page references, and retrieved excerpts.

## Suggested manual test

Create a text file named `sample.txt` with this content:

```text
The Acme Research team tested a solar-powered ventilation prototype in May 2025.
The prototype reduced average indoor temperature by 4 degrees Celsius during the test.
The test lasted 14 days and included measurements at three locations.
The report recommends a larger trial before commercial deployment.
```

Upload it and ask:

- What did the prototype reduce?
- How long did the test last?
- What does the report recommend?
- What was the prototype's effect on annual revenue?

The last question is deliberately unsupported by the sample. The system should say that it could not find enough information in the uploaded documents.

## Configuration

The `.env` file supports:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per question |
| `MAX_FILE_SIZE_MB` | `20` | Maximum size per uploaded file |

## Important limitations

- Scanned/image-only PDFs are not OCR-processed. They may produce no extractable text.
- TXT files and URL articles do not have PDF page numbers, so their page reference is shown as unavailable.
- Some websites block automated extraction or require JavaScript/login; URL ingestion may not work for those pages.
- The application uses in-memory FAISS indexing. The index is rebuilt when documents are processed and is not persisted after the app session ends.
- Retrieval can miss relevant passages if chunking or the retrieval count is unsuitable.
- LLM answers can still be incomplete or incorrectly interpret retrieved text. Always verify important claims against the displayed source excerpts.
- OpenAI API usage may incur charges. Embedding and chat requests send document text/question content to OpenAI.

## Security

- Keep `.env` private.
- `.env` is excluded through `.gitignore`.
- Never commit or share your real API key.
- If a key is accidentally exposed, revoke it and create a replacement.

## Project structure

```text
generative-ai-rag-system/
│
├── app.py
├── config.py
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   └── qa_chain.py
│
├── utils/
│   ├── __init__.py
│   └── helpers.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
The .env file is created locally by each user and is intentionally excluded from version control to protect API credentials
## Technologies

Python · LangChain · UnstructuredURLLoader · OpenAI Embeddings · FAISS · Streamlit · PyPDF
