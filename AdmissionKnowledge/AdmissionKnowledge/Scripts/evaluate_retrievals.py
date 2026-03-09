import json
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np



GROUND_TRUTH_FILE = r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\evaluation\ground_truth.json"

CHUNK_FILES = {
    "fixed": r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_fixed.json",
    "section": r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_section.json",
    "hybrid": r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_hybrid.json",
}

VECTOR_DB_PATHS = {
    "fixed": r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\vectorstores\chroma_fixed",
    "section": r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\vectorstores\chroma_section",
    "hybrid": r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\vectorstores\chroma_hybrid",
}

K_VALUES = [1, 3]
ALPHA = 0.5  # hybrid weight
RRF_K = 60


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / len(relevant) if relevant else 0


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / k


def reciprocal_rank(retrieved, relevant):
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / i
    return 0


def normalize(scores):
    scores = np.array(scores)
    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s == 0:
        return scores

    return (scores - min_s) / (max_s - min_s)


def build_bm25(chunks):

    corpus = [c["content"] for c in chunks]
    tokenized = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized)

    chunk_ids = [c["chunk_id"] for c in chunks]

    return bm25, chunk_ids, corpus


def bm25_search(bm25, chunk_ids, query, k=3):

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = np.argsort(scores)[::-1][:k]

    return [chunk_ids[i] for i in ranked_indices]


def semantic_search(collection, model, query, k=3):

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["ids"][0]


def hybrid_search(query, chunks, bm25, model):

    texts = [c["content"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    query_emb = model.encode(query)
    doc_embs = model.encode(texts)

    semantic_scores = np.dot(doc_embs, query_emb) / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
    )

    bm25_norm = normalize(bm25_scores)
    semantic_norm = normalize(semantic_scores)

    hybrid_scores = ALPHA * semantic_norm + (1 - ALPHA) * bm25_norm

    ranked_idx = np.argsort(hybrid_scores)[::-1]

    return [chunk_ids[i] for i in ranked_idx[:3]]


def rrf_search(query, chunks, bm25, collection, model):

    chunk_ids = [c["chunk_id"] for c in chunks]

    #  BM25 ranking 
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_idx = np.argsort(bm25_scores)[::-1]

    bm25_ranking = [chunk_ids[i] for i in bm25_ranked_idx]

    # Semantic ranking 
    query_emb = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=len(chunk_ids)
    )

    semantic_ranking = results["ids"][0]

    # RRF scoring 
    rrf_scores = {}

    for rank, doc_id in enumerate(bm25_ranking):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    for rank, doc_id in enumerate(semantic_ranking):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [doc[0] for doc in ranked[:3]]


def evaluate():

    ground_truth = load_ground_truth()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n\n==============================")
    print(" Retrieval Evaluation Results ")
    print("==============================")

    for strategy in ["fixed", "section", "hybrid"]:

        print(f"\nLoading {strategy} chunks...")

        chunks = load_chunks(CHUNK_FILES[strategy])

        bm25, chunk_ids, corpus = build_bm25(chunks)

        client = chromadb.PersistentClient(path=VECTOR_DB_PATHS[strategy])
        collection = client.get_collection("rag_collection")

        methods = {
             "bm25": [],
             "semantic": [],
             "hybrid": [],
             "rrf": []
       }

        for item in ground_truth:

            query = item["query"]
            relevant = item["relevant"][strategy]

            bm25_ids = bm25_search(bm25, chunk_ids, query)

            semantic_ids = semantic_search(collection, model, query)

            hybrid_ids = hybrid_search(query, chunks, bm25, model)

            rrf_ids = rrf_search(query, chunks, bm25, collection, model)

            methods["bm25"].append((bm25_ids, relevant))
            methods["semantic"].append((semantic_ids, relevant))
            methods["hybrid"].append((hybrid_ids, relevant))
            methods["rrf"].append((rrf_ids, relevant))

        compute_metrics(methods)


def compute_metrics(methods):

    for method in methods:

        recalls_1 = []
        recalls_3 = []

        precs_1 = []
        precs_3 = []

        mrrs = []

        for retrieved, relevant in methods[method]:

            recalls_1.append(recall_at_k(retrieved, relevant, 1))
            recalls_3.append(recall_at_k(retrieved, relevant, 3))

            precs_1.append(precision_at_k(retrieved, relevant, 1))
            precs_3.append(precision_at_k(retrieved, relevant, 3))

            mrrs.append(reciprocal_rank(retrieved, relevant))

        print(f"\nRetrieval Method: {method}")

        print(f"Recall@1:    {np.mean(recalls_1):.3f}")
        print(f"Recall@3:    {np.mean(recalls_3):.3f}")

        print(f"Precision@1: {np.mean(precs_1):.3f}")
        print(f"Precision@3: {np.mean(precs_3):.3f}")

        print(f"MRR:         {np.mean(mrrs):.3f}")


if __name__ == "__main__":
    evaluate()