
import os
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from embedding_service import generate_embeddings
from gemini_service import rewrite_search_query


# ==========================================
# GLOBAL CACHE
# ==========================================

cached_index = None
cached_chunks = None
cached_bm25 = None


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

INDEX_PATH = os.path.join(VECTOR_DIR, "company.index")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.json")


# ==========================================
# TEXT TOKENIZATION
# ==========================================

def tokenize(text):
    """
    Convert text into lowercase tokens
    for BM25 keyword retrieval.
    """

    text = text.lower()

    return re.findall(r"\b[a-z0-9]+\b", text)


# ==========================================
# EXPERIENCE-AWARE RANKING
# ==========================================

def get_experience_boost(query, chunk):
    """
    Give a small ranking preference to document
    passages that match the requested experience type.

    This does not create new information.
    """

    query_tokens = set(tokenize(query))
    chunk_lower = chunk.lower()

    boost = 0.0

    # Data Science internship relevance
    internship_terms = {
        "intern",
        "internship",
        "interning"
    }

    if query_tokens.intersection(internship_terms):

        if (
            "data science intern" in chunk_lower
            or "internship" in chunk_tokens(chunk_lower)
        ):
            boost += 0.002

    # Python Full Stack training relevance
    training_terms = {
        "training",
        "trainee",
        "trained",
        "train"
    }

    if query_tokens.intersection(training_terms):

        if (
            "python full stack trainee" in chunk_lower
            or {
                "python",
                "full",
                "stack",
                "trainee"
            }.issubset(query_tokens)
        ):
            boost += 0.002

    return boost


def chunk_tokens(text):
    """
    Tokenize a document chunk.
    """

    return tokenize(text)


# ==========================================
# LOAD EXISTING VECTOR DATA
# ==========================================

def load_vector_data():
    """
    Load the saved FAISS index and document chunks.

    Build the BM25 index only once and keep all
    three objects in memory.
    """

    global cached_index
    global cached_chunks
    global cached_bm25

    if (
        cached_index is not None
        and cached_chunks is not None
        and cached_bm25 is not None
    ):
        return cached_index, cached_chunks, cached_bm25

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Please upload a PDF document first."
        )

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Document chunks not found. Please upload a PDF document first."
        )

    print("Loading saved FAISS index...")

    cached_index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        cached_chunks = json.load(file)

    if not cached_chunks:
        raise ValueError("No document chunks are available.")

    if cached_index.ntotal != len(cached_chunks):
        raise RuntimeError(
            "FAISS index and document chunks do not match. "
            "Please upload the PDF again."
        )

    print("Building BM25 keyword index...")

    tokenized_chunks = [
        tokenize(chunk)
        for chunk in cached_chunks
    ]

    cached_bm25 = BM25Okapi(tokenized_chunks)

    print("FAISS and BM25 indexes loaded successfully.")

    return cached_index, cached_chunks, cached_bm25


# ==========================================
# CREATE VECTOR STORE
# ==========================================

def create_vector_store(chunks):
    """
    Generate embeddings for document chunks,
    create the FAISS index, and save the chunks.
    Also initialize the BM25 index.
    """

    global cached_index
    global cached_chunks
    global cached_bm25

    if not chunks:
        raise ValueError("Cannot create a vector store from empty chunks.")

    # Remove empty chunks
    clean_chunks = [
        str(chunk).strip()
        for chunk in chunks
        if chunk and str(chunk).strip()
    ]

    if not clean_chunks:
        raise ValueError("No valid document text was found.")

    print(f"Generating embeddings for {len(clean_chunks)} chunks...")

    embeddings = generate_embeddings(clean_chunks)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(clean_chunks):
        raise ValueError("Invalid embedding dimensions.")

    # Normalize embeddings for cosine similarity through inner product
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    print("Creating FAISS index...")

    new_index = faiss.IndexFlatIP(dimension)

    new_index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    faiss.write_index(
        new_index,
        INDEX_PATH
    )

    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            clean_chunks,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Build BM25 index
    tokenized_chunks = [
        tokenize(chunk)
        for chunk in clean_chunks
    ]

    new_bm25 = BM25Okapi(tokenized_chunks)

    # Update global cache
    cached_index = new_index
    cached_chunks = clean_chunks
    cached_bm25 = new_bm25

    print("FAISS and BM25 indexes created successfully.")

    return True


# ==========================================
# HYBRID DOCUMENT SEARCH
# ==========================================

def search_documents(query, top_k=3):
    """
    Hybrid retrieval pipeline:

    1. Validate original question.
    2. Use Gemini to understand and reformulate the question.
    3. Search the original and reformulated query together.
    4. Perform semantic retrieval using FAISS.
    5. Perform keyword retrieval using BM25.
    6. Combine results using Reciprocal Rank Fusion.
    7. Apply experience-aware ranking.
    8. Return the best matching document chunks.

    The original user question is preserved for ranking.
    """

    global cached_index
    global cached_chunks
    global cached_bm25

    # --------------------------------------
    # INPUT VALIDATION
    # --------------------------------------

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # --------------------------------------
    # LOAD DOCUMENT INDEXES
    # --------------------------------------

    index, chunks, bm25 = load_vector_data()

    # --------------------------------------
    # GEMINI QUERY UNDERSTANDING
    # --------------------------------------

    original_query = query.strip()

    print("Understanding the user question with Gemini...")

    rewritten_query = rewrite_search_query(original_query)

    if not rewritten_query or not rewritten_query.strip():
        rewritten_query = original_query

    print(f"Original query: {original_query}")
    print(f"Reformulated query: {rewritten_query}")

    # Combine original and reformulated questions.
    # This preserves the original terms while adding
    # semantic context for retrieval.

    retrieval_query = (
        original_query + " " + rewritten_query.strip()
    )

    # --------------------------------------
    # SEMANTIC SEARCH - FAISS
    # --------------------------------------

    print("Performing semantic retrieval...")

    query_embedding = generate_embeddings(
        [retrieval_query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    total_documents = index.ntotal

    candidate_k = min(
        max(top_k * 3, 10),
        total_documents
    )

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        candidate_k
    )

    # Store semantic scores and ranks
    semantic_rank = {}
    semantic_score_map = {}

    for rank, (score, index_id) in enumerate(
        zip(
            semantic_scores[0],
            semantic_indices[0]
        )
    ):

        if index_id < 0:
            continue

        index_id = int(index_id)

        semantic_rank[index_id] = rank

        semantic_score_map[index_id] = float(score)

    # --------------------------------------
    # KEYWORD SEARCH - BM25
    # --------------------------------------

    print("Performing BM25 keyword retrieval...")

    bm25_tokens = tokenize(retrieval_query)

    bm25_scores = bm25.get_scores(bm25_tokens)

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: float(bm25_scores[i]),
        reverse=True
    )

    # Exclude documents with no positive keyword match
    bm25_ranked_indices = [
        index_id
        for index_id in bm25_ranked_indices
        if bm25_scores[index_id] > 0
    ][:candidate_k]

    bm25_rank = {
        index_id: rank
        for rank, index_id in enumerate(bm25_ranked_indices)
    }

    # --------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------

    print("Combining retrieval results using RRF...")

    rrf_constant = 60

    hybrid_scores = {}

    candidate_indices = (
        set(semantic_rank.keys())
        | set(bm25_rank.keys())
    )

    for index_id in candidate_indices:

        hybrid_score = 0.0

        # Semantic retrieval contribution
        if index_id in semantic_rank:

            hybrid_score += 1.0 / (
                rrf_constant + semantic_rank[index_id] + 1
            )

        # BM25 retrieval contribution
        if index_id in bm25_rank:

            hybrid_score += 1.0 / (
                rrf_constant + bm25_rank[index_id] + 1
            )

        hybrid_scores[index_id] = hybrid_score

    # --------------------------------------
    # EXPERIENCE-AWARE RANKING
    # --------------------------------------

    print("Applying experience-aware ranking...")

    results = []

    for index_id in candidate_indices:

        chunk = chunks[index_id]

        hybrid_score = hybrid_scores[index_id]

        # Use ORIGINAL user query here.
        experience_boost = get_experience_boost(
            original_query,
            chunk
        )

        ranking_score = (
            hybrid_score + experience_boost
        )

        if (
            index_id in semantic_rank
            and index_id in bm25_rank
        ):
            matched_by = "semantic+keyword"

        elif index_id in semantic_rank:
            matched_by = "semantic"

        else:
            matched_by = "keyword"

        results.append(
            {
                "chunk": chunk,
                "score": float(hybrid_score),
                "ranking_score": float(ranking_score),
                "semantic_score": semantic_score_map.get(
                    index_id,
                    0.0
                ),
                "bm25_score": float(
                    bm25_scores[index_id]
                ),
                "matched_by": matched_by
            }
        )

    # --------------------------------------
    # FINAL RANKING
    # --------------------------------------

    results.sort(
        key=lambda item: item["ranking_score"],
        reverse=True
    )

    final_results = results[:top_k]

    print(
        f"Hybrid search completed. "
        f"Returned {len(final_results)} document chunks."
    )

    return final_results
