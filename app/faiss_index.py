import numpy as np
import faiss

from text_splitter import split_text
from embeddings import get_embeddings


def build_faiss_index(vectors):
    """
    Build a FAISS index from embedding vectors.
    We normalize vectors so we can use cosine-like similarity.
    """
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product after normalization
    index.add(vectors)

    return index


if __name__ == "__main__":
    # Load text
    with open("data/output.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    chunks = split_text(text)

    # Get embeddings
    vectors = get_embeddings(chunks)

    # Build FAISS index
    index = build_faiss_index(vectors)

    print("FAISS index built successfully!")
    print("Total vectors in index:", index.ntotal)

    # Test query
    query = "What is this document about?"
    query_vector = get_embeddings([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    k = 3
    distances, indices = index.search(query_vector, k)

    print("\nTop matching chunks:\n")
    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}:")
        print("Score:", distances[0][i])
        print(chunks[idx][:400])
        print("-" * 50)