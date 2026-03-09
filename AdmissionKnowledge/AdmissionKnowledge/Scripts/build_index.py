import os
import json
import argparse
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def load_chunks(chunk_file):
    with open(chunk_file, "r", encoding="utf-8") as f:
        return json.load(f)


def reset_collection(client, collection_name):
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(name=collection_name)

    print(f"Creating new collection: {collection_name}")
    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )


def build_index(chunk_file, db_path, collection_name="rag_collection"):
    print("Loading chunks...")
    chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Initializing Chroma client...")
    client = chromadb.PersistentClient(path=db_path)

    collection = reset_collection(client, collection_name)

    documents = []
    metadatas = []
    ids = []

    print("Preparing data for embedding...")

    for chunk in chunks:
        documents.append(chunk["content"])
        metadatas.append(chunk["metadata"])
        if "chunk_id" not in chunk:
            raise ValueError("chunk_id missing in JSON. Run add_chunk_ids.py first.")
        ids.append(chunk["chunk_id"])

    print("Generating embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Adding to Chroma collection...")
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

    print("Indexing complete!")
    print(f"Database stored at: {db_path}")
    print(f"Total indexed chunks: {len(documents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma index from chunk JSON file")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSON file")
    parser.add_argument("--db", required=True, help="Path to Chroma DB directory")
    parser.add_argument("--collection", default="rag_collection", help="Collection name")

    args = parser.parse_args()

    os.makedirs(args.db, exist_ok=True)

    build_index(
        chunk_file=args.chunks,
        db_path=args.db,
        collection_name=args.collection
    )