from rag_pipeline import RAGPipeline


rag = RAGPipeline()

print("\n Hello! I'm UniMind, your Data Science Admission Assistant! How can I help you today?\n")

while True:

    query = input("Ask a question: ")

    answer, chunks = rag.generate_answer(query)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")

    seen = set()

    for chunk in chunks:

        url = chunk["metadata"]["url"]
        title = chunk["metadata"]["title"]

        if url not in seen:

            print(f"• {title}")
            print(url)
            print()

            seen.add(url)