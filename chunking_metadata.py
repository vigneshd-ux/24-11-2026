"""A friendly, runnable demo of chunking, citations, and stable chunk IDs.

The script creates five tiny source documents, reads them back with the requested
libraries, compares chunking approaches, and simulates an incremental vector index.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_documents"
NAMESPACE = uuid.UUID("4f8d52b2-34bb-4f9b-a849-62a4d60c3a17")


@dataclass
class Page:
    """One page of text, with the heading we found for that page."""

    source: str
    page: int
    heading: str
    text: str
    doc_type: str


@dataclass
class Chunk:
    """Text plus enough context to cite it and recognize it again later."""

    chunk_id: str
    text: str
    metadata: dict


def make_paragraph(first_line: str, topic: str) -> str:
    """Build a readable sample paragraph long enough to create several chunks."""
    supporting_lines = {
        "garden": "Garden volunteers record rainfall, check the soil, and clear leaves from the inlet after each storm.",
        "health": "The clinic team records each visit, checks the equipment, and explains the next step to every patient.",
        "library": "Library staff count visitors, label new books, and help neighbors find reliable local information.",
        "energy": "The building team reads the meter, checks the panels, and compares the results with last month.",
        "river": "River stewards measure the water, note wildlife, and share their observations with the local community.",
    }[topic]
    # A few repetitions keep the demo documents substantial without adding dependencies.
    return first_line + " " + " ".join([supporting_lines] * 18)


DOCUMENTS = [
    (
        "community_health.pdf",
        "pdf",
        [
            ("Clinic opening hours", make_paragraph("The community clinic opens at 8 AM on weekdays.", "health")),
            ("Vaccine storage", make_paragraph("The vaccine refrigerator must stay at 5 degrees Celsius.", "health")),
            ("Emergency contact", make_paragraph("The after-hours nurse can be reached through the city hotline.", "health")),
        ],
    ),
    (
        "urban_gardens.pdf",
        "pdf",
        [
            ("Rain garden test", make_paragraph("The rain garden absorbed 42 liters of runoff during the storm.", "garden")),
            ("Compost schedule", make_paragraph("The neighborhood compost is turned every Saturday morning.", "garden")),
            ("Native planting", make_paragraph("Native flowers provide food for bees throughout the summer.", "garden")),
        ],
    ),
    (
        "library_guide.docx",
        "docx",
        [
            ("Borrowing books", make_paragraph("Library members may borrow 12 books for three weeks.", "library")),
            ("Quiet study room", make_paragraph("The quiet study room is reserved online until 6 PM.", "library")),
            ("Digital collection", make_paragraph("The library's digital collection is available every day.", "library")),
        ],
    ),
    (
        "building_energy.docx",
        "docx",
        [
            ("Solar panel output", make_paragraph("The roof panels produced 86 kilowatt hours on Monday.", "energy")),
            ("Heating schedule", make_paragraph("The office heating switches off automatically at 7 PM.", "energy")),
            ("Lighting audit", make_paragraph("Motion sensors turn off meeting room lights when rooms are empty.", "energy")),
        ],
    ),
    (
        "river_field_notes.txt",
        "txt",
        [
            ("Water quality", make_paragraph("The north river sample measured 7.2 on the pH scale.", "river")),
            ("Bird survey", make_paragraph("Volunteers counted 34 herons beside the river on Tuesday.", "river")),
            ("River flow", make_paragraph("The river gauge showed a steady flow after the spring rain.", "river")),
        ],
    ),
]


def create_sample_documents() -> None:
    """Create our five sample files so the demo works from a clean folder."""
    SAMPLE_DIR.mkdir(exist_ok=True)
    for filename, kind, pages in DOCUMENTS:
        path = SAMPLE_DIR / filename
        if kind == "pdf":
            pdf = fitz.open()
            for heading, text in pages:
                page = pdf.new_page()
                page.insert_textbox(fitz.Rect(48, 48, 548, 790), f"{heading}\n\n{text}", fontsize=10)
            pdf.save(path)
            pdf.close()
        elif kind == "docx":
            document = Document()
            for page_number, (heading, text) in enumerate(pages):
                if page_number:
                    document.add_page_break()
                document.add_heading(heading, level=1)
                document.add_paragraph(text)
            document.save(path)
        else:
            path.write_text(
                "\n\f\n".join(f"# {heading}\n\n{text}" for heading, text in pages),
                encoding="utf-8",
            )


def heading_and_body(text: str) -> tuple[str, str]:
    """Read a simple first-line heading from a page or return a sensible fallback."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith("#"):
        return lines[0].lstrip("# "), "\n".join(lines[1:])
    if lines:
        return lines[0], "\n".join(lines[1:])
    return "Untitled section", ""


def read_pdf(path: Path) -> list[Page]:
    pages = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            heading, body = heading_and_body(page.get_text())
            pages.append(Page(path.name, page_number, heading, body, "pdf"))
    return pages


def read_docx(path: Path) -> list[Page]:
    """Read Word paragraphs and use explicit page breaks as page boundaries."""
    document = Document(path)
    pages: list[Page] = []
    page_number = 1
    heading = "Untitled section"
    body: list[str] = []

    def save_page() -> None:
        pages.append(Page(path.name, page_number, heading, "\n".join(body), "docx"))

    for paragraph in document.paragraphs:
        if paragraph.paragraph_format.page_break_before and body:
            save_page()
            page_number += 1
            heading, body = "Untitled section", []
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading"):
            if body:
                save_page()
                page_number += 1
                body = []
            heading = text
        else:
            body.append(text)
    if body or not pages:
        save_page()
    return pages


def read_txt(path: Path) -> list[Page]:
    """Plain text uses form-feed characters as explicit page breaks."""
    pages = []
    for page_number, raw_page in enumerate(path.read_text(encoding="utf-8").split("\f"), start=1):
        heading, body = heading_and_body(raw_page)
        pages.append(Page(path.name, page_number, heading, body, "txt"))
    return pages


def read_all_documents() -> list[Page]:
    pages: list[Page] = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            pages.extend(read_pdf(path))
        elif path.suffix.lower() == ".docx":
            pages.extend(read_docx(path))
        elif path.suffix.lower() == ".txt":
            pages.extend(read_txt(path))
    return pages


def word_tokens(text: str) -> list[str]:
    """A lightweight, readable tokenizer for this teaching example."""
    return re.findall(r"\b[\w'-]+\b", text.lower())


def stable_id(page: Page, section: str, index: int, text: str) -> str:
    """Same location and content always produce the same UUID5 identifier."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = f"{page.source}|{page.page}|{section}|{index}|{content_hash}"
    return str(uuid.uuid5(NAMESPACE, identity))


def make_chunk(page: Page, section: str, index: int, text: str) -> Chunk:
    cleaned = text.strip()
    metadata = {
        "source": page.source,
        "page": page.page,
        "section_heading": section,
        "created_date": date.today().isoformat(),
        "chunk_index": index,
        "doc_type": page.doc_type,
    }
    return Chunk(stable_id(page, section, index, cleaned), cleaned, metadata)


def chunks_by_words(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Fixed-size token chunks, restarting at each page so citations stay clear."""
    chunks = []
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk size")
    for page in pages:
        tokens = word_tokens(page.text)
        for start in range(0, len(tokens), step):
            piece = tokens[start : start + size]
            if not piece:
                continue
            chunks.append(make_chunk(page, page.heading, len(chunks) + 1, " ".join(piece)))
            if start + size >= len(tokens):
                break
    return chunks


def sentence_chunks(pages: list[Page], max_words: int = 120) -> list[Chunk]:
    """Keep sentence boundaries, combining nearby sentences up to a word limit."""
    chunks = []
    for page in pages:
        sentences = re.split(r"(?<=[.!?])\s+", page.text.strip())
        current: list[str] = []
        current_size = 0
        for sentence in sentences:
            count = len(word_tokens(sentence))
            if current and current_size + count > max_words:
                chunks.append(make_chunk(page, page.heading, len(chunks) + 1, " ".join(current)))
                current, current_size = [], 0
            current.append(sentence)
            current_size += count
        if current:
            chunks.append(make_chunk(page, page.heading, len(chunks) + 1, " ".join(current)))
    return chunks


def recursive_chunks(pages: list[Page], size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Use LangChain's recursive splitter with word count as its size measure."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=lambda text: len(word_tokens(text)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            chunks.append(make_chunk(page, page.heading, len(chunks) + 1, piece))
    return chunks


def describe_chunk(chunk: Chunk) -> str:
    """Return a short citation suitable for search results or a report."""
    meta = chunk.metadata
    return (
        f"{meta['source']}, page {meta['page']}, {meta['section_heading']} "
        f"(chunk {meta['chunk_index']}; {meta['created_date']})"
    )


QUESTIONS = [
    ("When does the community clinic open?", "community_health.pdf", 1, "8 AM"),
    ("What temperature should the vaccine refrigerator stay at?", "community_health.pdf", 2, "5 degrees Celsius"),
    ("How much runoff did the rain garden absorb?", "urban_gardens.pdf", 1, "42 liters"),
    ("When is the neighborhood compost turned?", "urban_gardens.pdf", 2, "Saturday morning"),
    ("How many books can members borrow?", "library_guide.docx", 1, "12 books"),
    ("Until when is the quiet study room reserved?", "library_guide.docx", 2, "6 PM"),
    ("How much energy did the roof panels produce?", "building_energy.docx", 1, "86 kilowatt hours"),
    ("When does the office heating switch off?", "building_energy.docx", 2, "7 PM"),
    ("What pH did the north river sample measure?", "river_field_notes.txt", 1, "7.2"),
    ("How many herons did volunteers count?", "river_field_notes.txt", 2, "34 herons"),
]


def hit_rate(chunks: list[Chunk]) -> tuple[int, int]:
    """Top-one retrieval accuracy: did the best chunk come from the known page?"""
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    document_vectors = vectorizer.fit_transform(texts)
    hits = 0
    for question, source, page_number, _answer in QUESTIONS:
        query_vector = vectorizer.transform([question])
        scores = cosine_similarity(query_vector, document_vectors)[0]
        best = chunks[int(scores.argmax())]
        if best.metadata["source"] == source and best.metadata["page"] == page_number:
            hits += 1
    return hits, len(QUESTIONS)


def show_chunk_size_experiment(pages: list[Page]) -> None:
    print("\nTen check questions (their source pages are the answer key):")
    for number, (question, source, page_number, _answer) in enumerate(QUESTIONS, start=1):
        print(f"  Q{number}: {question} -> {source}, page {page_number}")
    print("\nChunk-size experiment (top-1 known-page hit rate):")
    print("  Size | Chunks | Hits | Hit rate")
    for size in (128, 256, 512, 1024):
        chunks = chunks_by_words(pages, size=size, overlap=50)
        hits, total = hit_rate(chunks)
        print(f"  {size:>4} | {len(chunks):>6} | {hits:>4}/{total} | {hits / total:>7.0%}")


def replace_sample_paragraph() -> None:
    """Edit one whole paragraph to show that only its own chunks get new IDs."""
    path = SAMPLE_DIR / "urban_gardens.pdf"
    pages = read_pdf(path)
    original = pages[0].text
    edited = original.replace("42 liters of runoff during the storm", "48 liters of runoff during the storm")
    # Keep the example edit small and controlled; paragraph boundaries remain the same.
    if original == edited:
        raise RuntimeError("The sample paragraph to edit was not found.")
    pdf = fitz.open(path)
    page = pdf[0]
    page.clean_contents()
    page.draw_rect(fitz.Rect(0, 0, 612, 792), color=None, fill=(1, 1, 1), overlay=True)
    new_text = pages[0].heading + "\n\n" + edited
    page.insert_textbox(fitz.Rect(48, 48, 548, 790), new_text, fontsize=10)
    temporary = path.with_suffix(".tmp.pdf")
    pdf.save(temporary)
    pdf.close()
    temporary.replace(path)


def show_incremental_index_demo(pages: list[Page]) -> None:
    """Index, repeat the index, edit one page, and count changed chunk IDs."""
    index: dict[str, Chunk] = {}
    first_run = chunks_by_words(pages, size=128, overlap=50)
    index.update({chunk.chunk_id: chunk for chunk in first_run})
    first_count = len(index)

    same_run = chunks_by_words(read_all_documents(), size=128, overlap=50)
    index.update({chunk.chunk_id: chunk for chunk in same_run})
    second_count = len(index)

    old_ids = set(index)
    replace_sample_paragraph()
    edited_pages = read_all_documents()
    edited_run = chunks_by_words(edited_pages, size=128, overlap=50)
    new_ids = {chunk.chunk_id for chunk in edited_run}
    removed_ids = old_ids - new_ids
    added_ids = new_ids - old_ids
    replaced = len(removed_ids)
    new_by_id = {chunk.chunk_id: chunk for chunk in edited_run}
    changed_locations = {
        (index[chunk_id].metadata["source"], index[chunk_id].metadata["page"])
        for chunk_id in removed_ids
    } | {
        (new_by_id[chunk_id].metadata["source"], new_by_id[chunk_id].metadata["page"])
        for chunk_id in added_ids
    }
    index = {chunk.chunk_id: chunk for chunk in edited_run}

    print("\nStable IDs and incremental indexing:")
    print(f"  Run 1: {first_count} points")
    print(f"  Run 2: {second_count} points (re-indexed identical content)")
    print(f"  Run 3 (one edited page): {len(index)} points, {replaced} replaced")
    print("  Vector count stays steady because unchanged chunk IDs are updated in place.")
    print(f"  Changed location(s): {', '.join(f'{source} p.{page}' for source, page in sorted(changed_locations))}")


def main() -> None:
    create_sample_documents()
    pages = read_all_documents()
    print(f"Read {len({page.source for page in pages})} documents and {len(pages)} pages:")
    for suffix in (".pdf", ".docx", ".txt"):
        count = sum(1 for page in pages if Path(page.source).suffix == suffix)
        print(f"  {suffix}: {count} pages")

    fixed = chunks_by_words(pages, size=512, overlap=50)
    sentence = sentence_chunks(pages, max_words=120)
    recursive = recursive_chunks(pages, size=512, overlap=50)
    print("\nChunk counts (fixed size is measured in words/tokens):")
    print(f"  Fixed 512/50: {len(fixed)} chunks")
    print(f"  Sentence based (up to 120 words): {len(sentence)} chunks")
    print(f"  Recursive 512/50: {len(recursive)} chunks")

    # This example deliberately splits a sentence at the fixed-size boundary.
    demo_text = "A sentence can be split across a chunk boundary when its length is greater than the remaining space."
    short_piece = " ".join(word_tokens(demo_text)[:9])
    next_piece = " ".join(word_tokens(demo_text)[9:])
    print("\nSentence cut in half example (fixed chunks do not protect sentence boundaries):")
    print(f"  Chunk A ends: ...{short_piece}")
    print(f"  Chunk B starts: {next_piece}...")

    if fixed:
        print("\nExample citation:")
        print(f"  {describe_chunk(fixed[0])}")
        print(f"  Metadata fields: {', '.join(fixed[0].metadata)}")

    show_chunk_size_experiment(pages)
    show_incremental_index_demo(pages)


if __name__ == "__main__":
    main()
