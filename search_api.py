"""A small local semantic-search API with simple, readable building blocks."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Callable

import fitz
import numpy as np
from docx import Document
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.getenv("SEARCH_API_KEY", "")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = None
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class SearchHit(BaseModel):
    source: str
    page: int
    section: str
    score: float
    text: str


class SearchResponse(BaseModel):
    results: list[SearchHit]


class DocumentRecord:
    def __init__(self, source: str, page: int, section: str, text: str, doc_type: str):
        self.source = source
        self.page = page
        self.section = section
        self.text = text
        self.doc_type = doc_type.lower()
        self.vector: np.ndarray | None = None


def check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require X-API-Key on operational endpoints; health stays public."""
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def extract_document(filename: str, raw: bytes) -> list[DocumentRecord]:
    """Extract page text from PDF, DOCX, or TXT uploads."""
    suffix = Path(filename).suffix.lower()
    records: list[DocumentRecord] = []
    if suffix == ".pdf":
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    section = lines[0] if lines else "Untitled section"
                    records.append(DocumentRecord(filename, page_number, section, text, "pdf"))
    elif suffix == ".docx":
        document = Document(BytesIO(raw))
        page_number, section, body = 1, "Untitled section", []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style.name.startswith("Heading"):
                if body:
                    records.append(DocumentRecord(filename, page_number, section, " ".join(body), "docx"))
                    page_number += 1
                    body = []
                section = text
            else:
                body.append(text)
        if body:
            records.append(DocumentRecord(filename, page_number, section, " ".join(body), "docx"))
    elif suffix in {".txt", ".md"}:
        text = raw.decode("utf-8", errors="replace")
        for page_number, page_text in enumerate(text.split("\f"), start=1):
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            if not lines:
                continue
            section = lines[0].lstrip("# ")
            records.append(DocumentRecord(filename, page_number, section, " ".join(lines), "txt"))
    else:
        raise HTTPException(status_code=415, detail="Upload a PDF, DOCX, TXT, or Markdown file")
    if not records:
        raise HTTPException(status_code=422, detail="No readable text found in the uploaded document")
    return records


def create_app(model_loader: Callable[[], object] | None = None) -> FastAPI:
    """Build the app; tests can inject a tiny deterministic model."""
    app = FastAPI(title="Local Search API", version="1.0.0")
    app.state.model = None
    app.state.records = []

    def load_model():
        if model_loader is not None:
            return model_loader()
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # One model instance per app process, shared by uploads and queries.
        app.state.model = load_model()
        yield

    app.router.lifespan_context = lifespan

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.model is not None, "documents": len(app.state.records)}

    @app.post("/documents", dependencies=[Depends(check_api_key)])
    async def upload_document(file: UploadFile = File(...)):
        filename = Path(file.filename or "upload.txt").name
        records = extract_document(filename, await file.read())
        vectors = app.state.model.encode([record.text for record in records], convert_to_numpy=True)
        for record, vector in zip(records, vectors):
            record.vector = np.asarray(vector, dtype=float)
        # Re-uploading the same filename replaces that source rather than duplicating it.
        app.state.records = [r for r in app.state.records if r.source != filename] + records
        return {"source": filename, "pages_indexed": len(records), "total_chunks": len(app.state.records)}

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(check_api_key)])
    def search(request: SearchRequest):
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Embedding model is still loading")
        candidates = [r for r in app.state.records if request.doc_type is None or r.doc_type == request.doc_type.lower()]
        if not candidates:
            return SearchResponse(results=[])
        query_vector = np.asarray(app.state.model.encode([request.query], convert_to_numpy=True)[0], dtype=float)
        matrix = np.vstack([r.vector for r in candidates])
        scores = cosine_similarity([query_vector], matrix)[0]
        ranked = sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)
        hits = []
        for score, record in ranked:
            score = float(score)
            if request.min_score is not None and score < request.min_score:
                continue
            hits.append(SearchHit(source=record.source, page=record.page, section=record.section,
                                  score=score, text=record.text))
            if len(hits) == request.top_k:
                break
        return SearchResponse(results=hits)

    return app


app = create_app()
