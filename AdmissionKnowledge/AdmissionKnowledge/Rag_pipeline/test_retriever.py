from Hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

while True:

    query = input("\nAsk a question: ")

    chunks = retriever.retrieve(query)

    for i, c in enumerate(chunks, 1):
        print("\n--- Result", i, "---")
        print(c["metadata"]["title"])
        print(c["metadata"]["url"])