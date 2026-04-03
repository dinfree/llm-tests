# Copilot Instructions for LLM Tests

## Quick Start

**Environment:** Python 3.14.x with `.venv` virtual environment  
**Main entry point:** `simple_model_test.py`

```bash
# Install dependencies
pip install -r requirements.txt

# Run the CLI
python simple_model_test.py
```

## Configuration

The application requires two environment variables in `.env`:
- `API_KEY`: Authentication token for the proxy server
- `BASE_URL`: OpenAI-compatible API endpoint (e.g., `http://localhost/v1`)

Model names are hardcoded in the script:
- Text tests: `"text"`
- Vision tests: `"vision"`
- Embeddings: `"text-embedding-nomic-embed-text-v1.5"`

## Architecture Overview

This is a CLI testing tool for OpenAI-compatible API proxies with three main test paths:

### 1. **Text Testing** (`run_text_test`)
- User provides a prompt
- LLM responds via streaming
- Metrics tracked: input/output tokens, TTFT, TPS

### 2. **Vision Testing** (`run_vision_test`)
- Uses `sample.jpg` (must exist in root)
- Encodes image as base64 data URL
- Sends image + prompt to vision model
- Same streaming + metrics output

### 3. **RAG Embedding Testing** (`run_embedding_test`)
- User uploads `.txt` or `.pdf` file
- Documents split with `RecursiveCharacterTextSplitter` (chunk_size=700, overlap=150)
- **Hybrid retrieval:** Dense (MMR via FAISS) + Sparse (BM25)
- Results fused using Reciprocal Rank Fusion (RRF, k=60, limit=6 docs)
- System prompt + retrieved context → LLM response
- PDF extraction uses "layout" mode to preserve table structure

## Key Utilities

- **`get_required_env(key)`** - Validates non-empty env vars; raises `ConfigError` if missing
- **`create_chat_model(model_name)`** - Creates ChatOpenAI client with streaming enabled
- **`create_embeddings_model(model_name)`** - Creates embeddings client; disables token context length checking (OpenAI proxy compatibility)
- **`stream_response(llm, messages)`** - Streams response, extracts text/usage metadata, calculates TTFT and TPS
- **`print_metrics()`** - Pretty-prints token counts and performance metrics

## Code Patterns

**Metadata Extraction:** The `_extract_text()` and `_extract_usage()` functions handle variable response formats:
- Text from `chunk.content` (string or list of dicts with type="text")
- Usage from `chunk.usage_metadata` or nested in `chunk.response_metadata`
- Falls back to word-count estimation if metrics unavailable

**PDF Handling:**
- Uses `extraction_mode="layout"` in PyPDFLoader (better for tables)
- Pre-filters empty pages since scanned PDFs may have blank extractions
- Warns user if extraction fails on some pages

**Document Deduplication:** `_doc_key()` creates stable keys using source, page, start_index, and content hash for RRF fusion

## Test Data

Place test files in repository root:
- `sample.jpg` - Required for vision test; must be JPEG
- `sample.txt`, `sample.pdf` - Example files for RAG testing

## Recommended MCP Servers

For enhanced development experience, consider configuring:
- **File System MCP** - For managing and analyzing test files (PDFs, images, text documents)
- **Web Fetch MCP** - For testing API proxy integration and monitoring HTTP endpoints
