import json
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNK_FILE = r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\chunk_outputs\chunks_hybrid.json"
VECTOR_DB = r"C:\Users\Akash Verma\AdmissionKnowledge\AdmissionKnowledge\vectorstores\chroma_hybrid"

TOP_K = 3
ALPHA = 0.5


class HybridRetriever:

    def __init__(self):

        print("Loading chunks...")

        with open(CHUNK_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.texts = [c["content"] for c in self.chunks]
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]

        print("Building BM25 index...")
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Connecting to Chroma DB...")
        client = chromadb.PersistentClient(path=VECTOR_DB)
        self.collection = client.get_collection("rag_collection")

    def normalize(self, scores):

        scores = np.array(scores)
        min_s = scores.min()
        max_s = scores.max()

        if max_s - min_s == 0:
            return scores

        return (scores - min_s) / (max_s - min_s)

    def retrieve(self, query):

        tokenized_query = query.lower().split()

        # BM25
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Semantic
        query_emb = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=len(self.chunks)
        )

        semantic_ranking = results["ids"][0]

        semantic_scores = np.zeros(len(self.chunks))

        for rank, cid in enumerate(semantic_ranking):
            idx = self.chunk_ids.index(cid)
            semantic_scores[idx] = len(self.chunks) - rank

        # Normalize
        bm25_norm = self.normalize(bm25_scores)
        sem_norm = self.normalize(semantic_scores)

        hybrid_scores = ALPHA * sem_norm + (1 - ALPHA) * bm25_norm

        ranked_idx = np.argsort(hybrid_scores)[::-1]

        top_chunks = []

        for i in ranked_idx[:TOP_K]:
            top_chunks.append(self.chunks[i])

        return top_chunks