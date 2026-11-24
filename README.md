# Embeddings and cosine similarity

This standalone demo embeds ten sentences with the local `all-MiniLM-L6-v2` model,
then demonstrates cosine similarity, semantic search, duplicate detection, and
K-Means clustering.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 embeddings_cosine.py
```

On macOS, the executable is often named `python3` rather than `python`. If you
use a virtual environment, activate it first; then use the same `python3` commands.

The first run downloads the local model. To include the API timing and dimension
comparison, set `OPENAI_API_KEY` before running; the script uses
`text-embedding-3-small`. Without a key, it prints that the API comparison was
skipped and still runs the local examples.

The reported local time measures encoding after the model is loaded. It excludes
model download and initialization. API time includes the network request, so the
comparison is illustrative rather than a hardware benchmark.

## Task 2: chunking, metadata, and stable IDs

Run the second demo with:

```bash
python3 -m pip install -r requirements.txt
python3 chunking_metadata.py
```

It creates two PDFs, two Word documents, and one plain-text document in
`sample_documents/`, extracts them with PyMuPDF, python-docx, and normal file
reading, then prints chunk counts, a ten-question known-page retrieval table,
citations, and an incremental-indexing demonstration. The chunk-size experiment
uses a simple TF-IDF search and reports top-one page hit rate; this keeps the
example lightweight and makes the measurement easy to inspect.

## Task 4: search API and evaluation

Install the added API dependencies with `python3 -m pip install -r requirements.txt`.
Set an API key in your terminal, then start the service:

```bash
export SEARCH_API_KEY='choose-a-local-demo-key'
uvicorn search_api:app --reload
```

`GET /health` is public. `POST /documents` accepts a PDF, DOCX, TXT, or Markdown
file as multipart form field `file`; `POST /search` expects JSON such as
`{"query":"When does the clinic open?","top_k":5,"doc_type":"pdf","min_score":0.2}`.
Send the configured key in the `X-API-Key` header to both POST endpoints. The
model is loaded once when the API starts. Re-uploading the same filename replaces
its records rather than adding duplicate records.

Evaluate two chunk sizes over 15 questions with known pages using:

```bash
python3 evaluate_search.py
```

The evaluation prints hit@1, hit@3, hit@5, MRR, and elapsed evaluation time for
both configurations, then chooses the one with higher hit@3 (breaking ties with
MRR). Scores vary with model version and the sample corpus.

Run the six API checks with `python3 -m pytest -q`.
