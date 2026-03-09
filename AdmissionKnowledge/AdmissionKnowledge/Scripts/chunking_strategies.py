import json
import os
from tqdm import tqdm
import tiktoken


INPUT_FILE = r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\Data\cleaned_updated_knowledge_base.json"

OUTPUT_FIXED = "chunks_fixed.json"
OUTPUT_SECTION = "chunks_section.json"
OUTPUT_HYBRID = "chunks_hybrid.json"

MODEL_NAME = "gpt-4o-mini"  # tokenizer reference
MAX_TOKENS = 600
OVERLAP = 100

# TOKENIZER


encoding = tiktoken.encoding_for_model(MODEL_NAME)


def count_tokens(text):
    return len(encoding.encode(text))


# STRATEGY 1
# FIXED TOKEN CHUNKING


def chunk_fixed(text):
    tokens = encoding.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + MAX_TOKENS
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += MAX_TOKENS - OVERLAP

    return chunks


# STRATEGY 2
# SECTION-BASED CHUNKING


def chunk_by_sections(text):
    # Split on double line breaks or sentence boundaries
    sections = text.split("  ")

    chunks = []
    current_chunk = ""

    for section in sections:
        if count_tokens(current_chunk + section) <= MAX_TOKENS:
            current_chunk += section + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# STRATEGY 3
# HYBRID CHUNKING


def chunk_hybrid(text):
    # First split by sections
    sections = text.split("  ")

    chunks = []

    for section in sections:
        if count_tokens(section) <= MAX_TOKENS:
            chunks.append(section.strip())
        else:
            # If too large → apply fixed chunking
            smaller_chunks = chunk_fixed(section)
            chunks.extend(smaller_chunks)

    return chunks


# METADATA HANDLING


def process_documents(chunk_function):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []

    for entry in tqdm(data):

        if entry["type"] == "deadlines":
            # Deadlines are structured — store as one chunk
            content = json.dumps(entry["deadlines"], indent=2)
            metadata = {
                "url": entry["url"],
                "title": "Deadlines",
                "type": "deadlines",
                "program": "Data Science (M.Sc.)"
            }

            all_chunks.append({
                "content": content,
                "metadata": metadata
            })
            continue

        text = entry.get("clean_content", "")
        if not text:
            continue

        chunks = chunk_function(text)

        for chunk in chunks:
            metadata = {
                "url": entry["url"],
                "title": entry.get("title", ""),
                "type": entry.get("type", "general"),
                "program": "Data Science (M.Sc.)"
            }

            all_chunks.append({
                "content": chunk,
                "metadata": metadata
            })

    return all_chunks


def main():

    print("Generating Fixed Chunks...")
    fixed_chunks = process_documents(chunk_fixed)
    with open(OUTPUT_FIXED, "w", encoding="utf-8") as f:
        json.dump(fixed_chunks, f, indent=2, ensure_ascii=False)

    print("Generating Section Chunks...")
    section_chunks = process_documents(chunk_by_sections)
    with open(OUTPUT_SECTION, "w", encoding="utf-8") as f:
        json.dump(section_chunks, f, indent=2, ensure_ascii=False)

    print("Generating Hybrid Chunks...")
    hybrid_chunks = process_documents(chunk_hybrid)
    with open(OUTPUT_HYBRID, "w", encoding="utf-8") as f:
        json.dump(hybrid_chunks, f, indent=2, ensure_ascii=False)

    print("All chunking strategies completed.")


if __name__ == "__main__":
    main()