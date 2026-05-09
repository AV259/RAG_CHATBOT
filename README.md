## RAG-Based Admission Query Chatbot
A domain-specific Retrieval-Augmented Generation (RAG) chatbot built to answer Master’s Data Science admission queries for Philipps University of Marburg using information collected from official university webpages.

The system combines web scraping, document preprocessing, semantic retrieval, and large language models to provide accurate, context-grounded responses while minimizing hallucinations.

Features: 
- End-to-end RAG pipeline for admission-related question answering
- Automated web scraping and structured data collection
- Multiple chunking strategy experiments for retrieval optimization
- Semantic search using MiniLM embeddings + ChromaDB
- Evaluation of BM25, semantic, hybrid, and RRF retrieval methods
- Grounded response generation using Qwen 2.5–7B
- Interactive chatbot deployment using Streamlit

# Demo Video

The Chatbot can be accessed here- https://unimind.streamlit.app/

https://github.com/user-attachments/assets/60b05d03-aac7-45b8-a650-efa023fa43b0



# Project Pipeline:

1. Data Collection
Admission-related webpages were scraped from the university website using Scrapy with:
 - Controlled link following
 - Ethical crawling practices
 - Domain restrictions
 - Structured metadata collection

2. Data Cleaning & Processing
The scraped data was cleaned and normalized by:
 - Removing navigation menus, footer text, and irrelevant HTML content
 - Classifying pages by content type
 - Applying structured preprocessing for deadline-related pages
 - Converting webpages into retrieval-ready text documents

3. Chunking Strategies

Three chunking strategies were designed and evaluated:
- Fixed Chunking: constant-size text chunks
- Section-Based Chunking: chunks aligned with webpage sections/headings
- Hybrid Chunking: combination of semantic structure and size constraints

The chunking approaches were compared to identify the most retrieval-friendly structure.

4. Embedding & Indexing

Processed chunks were embedded using:
- Sentence Transformers MiniLM embeddings
- Vector storage and retrieval were handled using ChromaDB

5. Retrieval Evaluation
The project evaluated four retrieval strategies:

- BM25 Retrieval
- Dense Semantic Retrieval
- Hybrid Retrieval
- Reciprocal Rank Fusion (RRF)

Performance was measured using:
- Precision
- Recall
- Mean Reciprocal Rank (MRR)

6. Response Generation
The best-performing retrieval setup was integrated with:
- Qwen 2.5–7B
- Hugging Face Inference API

A carefully engineered system prompt was used to:

- Keep answers grounded in retrieved context
- Reduce hallucinations
- Force abstention when information was unavailable

# Deployment 
The chatbot was deployed as an interactive web application using:
- Streamlit

This enables real-time admission query answering through a simple user interface.

# Tech Stack
- Data Collection
- Scrapy
- BeautifulSoup
- NLP & Retrieval
- Sentence Transformers (MiniLM)
- ChromaDB
- BM25
- Hybrid Retrieval / RRF
- LLM & Inference
- Qwen 2.5–7B
- Hugging Face Inference API
- Deployment
-Streamlit
-Hugging Face Spaces
