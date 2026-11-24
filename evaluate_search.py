"""Evaluate two chunk sizes on 15 questions with known source pages."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from search_api import extract_document


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_documents"
QUESTIONS = [
    ("When does the clinic open on weekdays?", "community_health.pdf", 1),
    ("What temperature should vaccine storage be?", "community_health.pdf", 2),
    ("How do I contact the after-hours nurse?", "community_health.pdf", 3),
    ("How much runoff does the rain garden absorb?", "urban_gardens.pdf", 1),
    ("When is compost turned?", "urban_gardens.pdf", 2),
    ("What do native flowers provide for bees?", "urban_gardens.pdf", 3),
    ("How many books can a library member borrow?", "library_guide.docx", 1),
    ("Until what time is the quiet study room reserved?", "library_guide.docx", 2),
    ("When is the digital collection available?", "library_guide.docx", 3),
    ("How much energy did the roof panels produce?", "building_energy.docx", 1),
    ("When does office heating switch off?", "building_energy.docx", 2),
    ("What makes meeting room lights switch off?", "building_energy.docx", 3),
    ("What pH did the north river sample measure?", "river_field_notes.txt", 1),
    ("How many herons were counted by volunteers?", "river_field_notes.txt", 2),
    ("What did the river gauge show after spring rain?", "river_field_notes.txt", 3),
]


def read_source_files():
    records = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            records.extend(extract_document(path.name, path.read_bytes()))
    return records


def make_chunks(records, size: int, overlap: int = 30):
    chunks = []
    step = size - overlap
    for record in records:
        words = re.findall(r"\S+", record.text)
        for start in range(0, len(words), step):
            text = " ".join(words[start : start + size])
            if text:
                chunks.append((record.source, record.page, text))
            if start + size >= len(words):
                break
    return chunks


def evaluate(model, chunks):
    doc_vectors = model.encode([chunk[2] for chunk in chunks], convert_to_numpy=True, show_progress_bar=False)
    query_vectors = model.encode([row[0] for row in QUESTIONS], convert_to_numpy=True, show_progress_bar=False)
    reciprocal_ranks = []
    hits = {1: 0, 3: 0, 5: 0}
    for query_vector, (_query, expected_source, expected_page) in zip(query_vectors, QUESTIONS):
        scores = cosine_similarity([query_vector], doc_vectors)[0]
        ranked_indices = np.argsort(scores)[::-1]
        rank = next((position for position, index in enumerate(ranked_indices, 1)
                     if chunks[index][0] == expected_source and chunks[index][1] == expected_page), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1
    total = len(QUESTIONS)
    return {f"hit@{k}": hits[k] / total for k in hits}, sum(reciprocal_ranks) / total


def main():
    # Create demo files first if Task 2 has not been run yet.
    from chunking_metadata import create_sample_documents

    create_sample_documents()
    records = read_source_files()
    model = SentenceTransformer("all-MiniLM-L6-v2")  # one model load for both runs
    results = []
    for label, size in (("A", 128), ("B", 256)):
        chunks = make_chunks(records, size)
        started = time.perf_counter()
        metrics, mrr = evaluate(model, chunks)
        elapsed = time.perf_counter() - started
        results.append((label, size, len(chunks), metrics, mrr, elapsed))
    print("Config | chunk words | chunks | hit@1 | hit@3 | hit@5 | MRR | eval seconds")
    for label, size, count, metrics, mrr, elapsed in results:
        print(f"{label:>6} | {size:>11} | {count:>6} | {metrics['hit@1']:.2f} | "
              f"{metrics['hit@3']:.2f} | {metrics['hit@5']:.2f} | {mrr:.2f} | {elapsed:.3f}")
    winner = max(results, key=lambda item: (item[3]["hit@3"], item[4]))
    print(f"Decision: Config {winner[0]} is better by hit@3, with MRR {winner[4]:.2f}.")
    print("The score is top-chunk page accuracy; this tiny demo is illustrative, not a production benchmark.")


if __name__ == "__main__":
    main()
