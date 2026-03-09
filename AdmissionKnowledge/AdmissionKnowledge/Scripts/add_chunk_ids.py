import json
import os


def add_ids_to_file(file_path, prefix):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n Processing: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{prefix}_{idx}"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Added {len(chunks)} chunk IDs with prefix '{prefix}'")



def process_fixed():
    add_ids_to_file(
        file_path=r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_fixed.json",
        prefix="fixed"
    )


def process_section():
    add_ids_to_file(
        file_path=r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_section.json",
        prefix="section"
    )


def process_hybrid():
    add_ids_to_file(
        file_path=r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_hybrid.json",
        prefix="hybrid"
    )


def main():
    print(" Adding chunk IDs to all strategies...")
    process_fixed()
    process_section()
    process_hybrid()
    print("\n All chunk files updated successfully.")


if __name__ == "__main__":
    main()