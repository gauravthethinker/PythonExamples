import os
import argparse
import json
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 256  # tokens/words per chunk approximation


def read_file(path: Path) -> str:
    """Read the entire file as UTF-8 text."""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """A naive whitespace-based chunking of the text.

    Splits the text into roughly `chunk_size` word chunks to keep embedding sizes manageable.
    """
    words = text.split()
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [c for c in chunks if c.strip()]


def build_or_load_index(dim: int, index_path: Path) -> faiss.IndexFlatL2:
    """Load an existing FAISS index from disk or create a new one."""
    if index_path.exists():
        print(f"Loading existing FAISS index from {index_path} …")
        return faiss.read_index(str(index_path))
    print("Creating new FAISS index …")
    return faiss.IndexFlatL2(dim)


def save_index(index: faiss.Index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def main():
    parser = argparse.ArgumentParser(description="Ingest a text file into a FAISS vector database.")
    parser.add_argument("input_file", type=Path, help="Path to the input text file to ingest.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name.")
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Approximate number of words per chunk."
    )
    parser.add_argument(
        "--index_path", type=Path, default=Path("vector_store/faiss.index"), help="Path to save/load the FAISS index."
    )
    parser.add_argument(
        "--metadata_path", type=Path, default=Path("vector_store/metadata.json"), help="Path to store chunk metadata."
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        parser.error(f"Input file {args.input_file} does not exist.")

    # Read and chunk the document
    text = read_file(args.input_file)
    chunks = chunk_text(text, args.chunk_size)
    print(f"Split input into {len(chunks)} chunks …")

    # Load embedding model
    print(f"Loading SentenceTransformer model '{args.model}' …")
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    # Build or load FAISS index
    index = build_or_load_index(dim, args.index_path)

    # Prepare metadata storage
    if args.metadata_path.exists():
        with args.metadata_path.open("r", encoding="utf-8") as f:
            metadata: List[str] = json.load(f)
    else:
        metadata = []

    # Embed and add chunks
    for chunk in chunks:
        embedding = model.encode(chunk)
        index.add(embedding.reshape(1, -1))
        metadata.append(chunk)

    print("Saving index and metadata …")
    save_index(index, args.index_path)
    with args.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Ingestion complete 🎉")


if __name__ == "__main__":
    main()