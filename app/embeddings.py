from sentence_transformers import SentenceTransformer

# Load a small, fast model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_list):
    """
    Convert a list of text chunks into embeddings.
    Returns a list of vectors.
    """
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    # Read chunks from previous step
    from text_splitter import split_text

    with open("data/output.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)
    vectors = get_embeddings(chunks)

    print("Total chunks:", len(chunks))
    print("Embedding shape:", vectors.shape)
    print("First embedding vector sample:")
    print(vectors[0][:10])