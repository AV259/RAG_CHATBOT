import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

FILES = [
    r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_fixed.json",
    r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_section.json",
    r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_hybrid.json"
]

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

for file in FILES:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_chunks = len(data)
    token_counts = [count_tokens(chunk["content"]) for chunk in data]

    avg_tokens = sum(token_counts) / total_chunks
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)

    print(f"\n=== {file} ===")
    print(f"Total chunks: {total_chunks}")
    print(f"Average tokens per chunk: {avg_tokens:.2f}")
    print(f"Max tokens in chunk: {max_tokens}")
    print(f"Min tokens in chunk: {min_tokens}")