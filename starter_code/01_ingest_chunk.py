"""
01_ingest_chunk.py — Module 4.2 / 4.3: Ingestion + Chunking

GOAL: Load all .txt files from your chosen scenario's data folder, split each into
overlapping chunks, and save the chunks (with metadata) to a JSON file for the next
stage to consume.

Expected output: outputs/chunks.json
    A list of {"id": ..., "source": ..., "text": ...} objects.

Time budget: ~15 minutes
"""

import os
import json
import glob
from config import DATA_DIR, SCENARIO

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "chunks.json")

# --- Tunable chunking parameters (Module 4.3) ---
CHUNK_SIZE = 400        # characters per chunk (simple fixed-size strategy)
CHUNK_OVERLAP = 80      # characters of overlap between consecutive chunks


def load_documents(data_dir: str) -> list[dict]:
    """Load every .txt file in data_dir. Returns [{"source": filename, "text": content}]."""
    docs = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(filepath), "text": text})
    return docs


def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    r"""
    Simple fixed-size chunking with overlap (Module 4.3).

    TODO: This is the beginner-tier implementation. For a stronger submission,
    also try (and compare in your RESULTS.md):
      - sentence-aware chunking (split on sentence boundaries, don't cut mid-sentence)
      - paragraph-aware chunking (split on blank lines first, then sub-chunk long paras)

    Hint for sentence-aware chunking:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # then greedily pack sentences into chunks up to chunk_size
    """
    import re
    chunks = []
    start = 0
    while start < len(text):
        text1=text[start:start+chunk_size]
        sentences = re.split(r'(?<=[.!?])\s+', text1)
        last_sentence=sentences[-1]
        if len(last_sentence)<chunk_size:
            if not re.search(r"\.\s+$",last_sentence):
                sentences=sentences[:-1]
                
        chunk = " ".join(sentences).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunk_records(docs: list[dict]) -> list[dict]:
    """Turn documents into a flat list of chunk records with stable IDs and metadata."""
    records = []
    chunk_id = 0
    for doc in docs:
        chunks = fixed_size_chunk(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_text in chunks:
            records.append({
                "id": f"chunk_{chunk_id:04d}",
                "source": doc["source"],
                "text": chunk_text,
            })
            chunk_id += 1
    return records


def main():
    print(f"Scenario: {SCENARIO}")
    print(f"Loading documents from: {DATA_DIR}")

    docs = load_documents(DATA_DIR)
    if not docs:
        raise RuntimeError(
            f"No .txt files found in {DATA_DIR}. Check SCENARIO in your .env "
            "(should be 'banking' or 'healthcare')."
        )
    print(f"Loaded {len(docs)} documents.")

    records = build_chunk_records(docs)
    print(f"Produced {len(records)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved chunks to: {OUTPUT_PATH}")

    # Quick sanity print
    print("\n--- Sample chunk ---")
    print(json.dumps(records[0], indent=2)[:500])


if __name__ == "__main__":
    main()
