import os
from huggingface_hub import InferenceClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Rag_pipeline.Hybrid_retriever import HybridRetriever


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN") 

system_prompt = """
You are UniMind, a helpful assistant for Philipps University of Marburg.
Answer questions about the Master's Data Science program using ONLY the provided context.
Rules:
- Base your answer strictly on the context. Do NOT add any information not present in the context.
- If the answer is not in the context, say you do not have that information.
- Do not mention "context" or "documents" in your answer.
- Be thorough: include all relevant details from the context.
- Be clear and well-structured.
""".strip()


class RAGPipeline:

    def __init__(self):

        print("Loading retriever...")
        self.retriever = HybridRetriever()

        print("Connecting to HuggingFace Inference API...")
        self.client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)


    def build_user_prompt(self, query, chunks):

        context_blocks = []

        for i, chunk in enumerate(chunks, 1):
            content = chunk["content"]
            context_blocks.append(f"[{i}] {content}")

        context = "\n\n".join(context_blocks)

        user_prompt = f"""Question: {query}

Context:
{context}

Answer the question above using only the context provided."""

        return user_prompt


    def generate_answer(self, query):

        # retrieve top chunks
        chunks = self.retriever.retrieve(query)

        # build prompt
        user_prompt = self.build_user_prompt(query, chunks)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.85,
        )

        answer = response.choices[0].message.content.strip()

        return answer, chunks
