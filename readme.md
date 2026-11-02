# Comparative Evaluation of Vector Databases and Hybrid Retrieval in RAG Systems

## Overview
This repository contains the reproducible benchmarking suite for evaluating local vector databases (FAISS, ChromaDB, Qdrant) and hybrid retrieval architectures (Dense + BM25 via Reciprocal Rank Fusion) within Retrieval-Augmented Generation (RAG) pipelines. 

This project isolates the vector storage engine by freezing document segments (~500 tokens) and pre-computing static BGE embeddings to prevent framework orchestration overhead from skewing metrics.

## Repository Structure
- `/notebooks`: Contains the Google Colab `.ipynb` notebooks for end-to-end execution.
- `/data/raw_pdfs`: 10 foundational AI/ML research papers used as the testing corpus.
- `/data/processed`: Evaluation queries, chunked JSONs, ground-truth baselines, and downstream LLM-as-a-Judge generation metrics.
- `/deliverables`: Plotted performance charts and the final Architectural Decision Matrix.

## Methodology
1. **Layer 1 (Backend Efficiency):** In-memory profiling of FAISS, ChromaDB, and Qdrant across latency, RAM, and exact-match Recall@k.
2. **Layer 2 (Hybrid Accuracy):** Fusing pure dense vector search with Okapi BM25 keyword matching using parameter-free RRF (k=60).
3. **Layer 3 (Generation Quality):** Automated evaluation of response faithfulness, relevance, and out-of-scope refusal accuracy using Gemini 3.5 Flash and Llama 3.3 70B.

## Execution
This suite is designed to run entirely within Google Colab's free-tier limits. Upload the contents of the `/data` directory to your Google Drive, mount the drive in the provided notebook, and execute the cells sequentially.