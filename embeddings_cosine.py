"""Embeddings, cosine similarity, semantic search, duplicate detection, and clustering.

Install dependencies with: python -m pip install -r requirements.txt
Set OPENAI_API_KEY to include the API comparison; the local examples always run.
"""

from __future__ import annotations

import os
import time
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


SENTENCES = [
    "Physician treats people who are ill.",
    "Doctor cares for sick patients.",
    "The clinic offers medical care to patients.",
    "A puppy plays happily with its owner.",
    "Dogs enjoy running and playing outside.",
    "The kitten chased a ball across the floor.",
    "A train carries commuters between cities.",
    "Many passengers travel to work by railway.",
    "The subway moves people around the city.",
    "Rail transport helps commuters reach work.",
]


def cosine_similarity(a, b) -> float:
    """Return cosine similarity, with a clear error for zero-length vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be one-dimensional vectors of equal length")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / denominator)


def api_embeddings(sentences: list[str]) -> tuple[np.ndarray, float]:
    """Embed texts using OpenAI's embedding API and return vectors and elapsed time."""
    from openai import OpenAI

    client = OpenAI()
    started = time.perf_counter()
    response = client.embeddings.create(model="text-embedding-3-small", input=sentences)
    elapsed = time.perf_counter() - started
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.asarray([item.embedding for item in ordered], dtype=float), elapsed


def print_matrix(matrix: np.ndarray, labels: list[str]) -> None:
    print("\nSimilarity matrix (local embeddings):")
    print("       " + " ".join(f"S{i + 1:>5}" for i in range(len(labels))))
    for i, row in enumerate(matrix):
        print(f"S{i + 1:>2}    " + " ".join(f"{value:6.2f}" for value in row))
    print("Sentence key:")
    for i, sentence in enumerate(labels, 1):
        print(f"  S{i}: {sentence}")


def main() -> None:
    print("Cosine similarity implementation checks:")
    examples = [([1, 2, 3], [2, 4, 6]), ([1, 0], [0, 1]), ([1, 0], [-1, 0])]
    for a, b in examples:
        own = cosine_similarity(a, b)
        reference = float(sklearn_cosine_similarity([a], [b])[0, 0])
        print(f"{a} vs {b} -> {own:.1f} (scikit-learn: {reference:.1f})")

    print("\nEmbedding 10 sentences with local all-MiniLM-L6-v2...")
    try:
        local_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        raise SystemExit(
            "Could not load all-MiniLM-L6-v2. On the first run, make sure you can "
            "reach Hugging Face so sentence-transformers can download the model. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc
    started = time.perf_counter()
    local_vectors = local_model.encode(SENTENCES, convert_to_numpy=True, show_progress_bar=False)
    local_seconds = time.perf_counter() - started
    print(f"Local: {local_vectors.shape[1]} dimensions, {local_seconds:.3f} seconds")

    if os.getenv("OPENAI_API_KEY"):
        try:
            remote_vectors, api_seconds = api_embeddings(SENTENCES)
            print(f"API (text-embedding-3-small): {remote_vectors.shape[1]} dimensions, {api_seconds:.3f} seconds")
            print(f"Timed API call was {api_seconds / local_seconds:.2f}x local encoding time (network and cold-start effects apply).")
        except Exception as exc:
            print(f"API embedding failed ({type(exc).__name__}: {exc}); continuing with local examples.")
    else:
        print("API: skipped (set OPENAI_API_KEY to compare text-embedding-3-small).")

    matrix = sklearn_cosine_similarity(local_vectors)
    print_matrix(matrix, SENTENCES)

    pairs = sorted(
        ((matrix[i, j], i, j) for i, j in combinations(range(len(SENTENCES)), 2)),
        reverse=True,
    )
    print("\nThree most similar pairs:")
    for score, i, j in pairs[:3]:
        print(f"  {score:.3f}: S{i + 1} / S{j + 1} — {SENTENCES[i]} | {SENTENCES[j]}")

    # The first two sentences use disjoint content words but express the same idea.
    no_common = cosine_similarity(local_vectors[0], local_vectors[1])
    print("\nNo-common-words example:")
    print(f'  "{SENTENCES[0]}" / "{SENTENCES[1]}" -> {no_common:.3f}')
    print("  (No shared words after lowercasing and removing punctuation.)")

    print("\nSemantic search (top match for each query):")
    queries = [
        "Where can someone get help when they are unwell?",
        "What animals like to play outdoors?",
        "How do people commute by rail?",
    ]
    query_vectors = local_model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
    for query, query_vector in zip(queries, query_vectors):
        scores = sklearn_cosine_similarity([query_vector], local_vectors)[0]
        best = int(np.argmax(scores))
        print(f'  "{query}" -> S{best + 1} ({scores[best]:.3f}): {SENTENCES[best]}')

    threshold = 0.72
    duplicates = [(score, i, j) for score, i, j in pairs if score >= threshold]
    print(f"\nDuplicate detection (threshold >= {threshold:.2f}):")
    if duplicates:
        for score, i, j in duplicates:
            print(f"  likely duplicate ({score:.3f}): S{i + 1} / S{j + 1}")
    else:
        print("  No pairs exceeded the threshold.")

    cluster_names = {0: "Healthcare", 1: "Pets and animals", 2: "Rail commuting"}
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(local_vectors)
    # Assign names based on the sentence with the highest cosine similarity to each centroid.
    center_matches = sklearn_cosine_similarity(kmeans.cluster_centers_, local_vectors)
    names_by_id = {}
    for cluster_id in range(3):
        member_indices = np.flatnonzero(cluster_ids == cluster_id)
        exemplar = member_indices[np.argmax(center_matches[cluster_id, member_indices])]
        text = SENTENCES[exemplar].lower()
        if any(word in text for word in ("doctor", "physician", "clinic", "patient", "medical")):
            names_by_id[cluster_id] = cluster_names[0]
        elif any(word in text for word in ("puppy", "dog", "kitten", "owner", "animal")):
            names_by_id[cluster_id] = cluster_names[1]
        else:
            names_by_id[cluster_id] = cluster_names[2]
    print("\nK-Means clusters (k=3):")
    for cluster_id in range(3):
        members = [f"S{i + 1}" for i, label in enumerate(cluster_ids) if label == cluster_id]
        print(f"  {names_by_id[cluster_id]}: {', '.join(members)}")

    print("\nChoosing local vs API (4 lines):")
    print("  Local models keep text on your machine and can work offline after download.")
    print("  They avoid per-request API charges, but use local memory and compute.")
    print("  APIs are quick to integrate and scale without managing model hardware.")
    print("  APIs send text to a provider and may add network latency and usage cost.")


if __name__ == "__main__":
    main()
