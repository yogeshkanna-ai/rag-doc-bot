import os
import re

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from app.text_splitter import split_text

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Put it inside the .env file.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=groq_api_key)

STOPWORDS = {
    "the", "is", "are", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "what", "which", "who", "whom", "this", "that", "it", "be", "as",
    "at", "by", "from", "about", "me", "my", "your", "their", "our", "its",
    "was", "were", "do", "does", "did", "tell", "show", "give", "can", "could"
}


def get_embeddings(text_list):
    return embedding_model.encode(text_list, show_progress_bar=False)


def build_faiss_index(vectors):
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def create_rag_data(text):
    chunks = split_text(text)
    vectors = get_embeddings(chunks)
    index = build_faiss_index(vectors)
    return chunks, index


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def keyword_overlap_score(question, chunk):
    q_words = tokenize(question)
    c_words = tokenize(chunk)

    if not q_words or not c_words:
        return 0.0

    overlap = len(q_words & c_words)
    return overlap / len(q_words)


def retrieve_top_chunks(question, chunks, index, k=8, candidate_k=20):
    query_vector = get_embeddings([question]).astype("float32")
    faiss.normalize_L2(query_vector)

    candidate_k = min(candidate_k, len(chunks))
    distances, indices = index.search(query_vector, candidate_k)

    scored_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        semantic_score = (float(dist) + 1.0) / 2.0
        keyword_score = keyword_overlap_score(question, chunks[idx])
        combined_score = (0.75 * semantic_score) + (0.25 * keyword_score)
        scored_chunks.append((combined_score, idx))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    top_chunks = []
    seen = set()
    for _, idx in scored_chunks:
        if idx not in seen:
            top_chunks.append(chunks[idx])
            seen.add(idx)
        if len(top_chunks) >= k:
            break

    return top_chunks


def ask_llm(question, top_chunks):
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a helpful document question-answering assistant.

Use ONLY the context below to answer the question as best as possible.
Do NOT refuse.
Do NOT say you could not find the answer.
Do NOT mention missing context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer only from the provided document context, but always give a best-effort answer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


def extract_email(text):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def extract_phone(text):
    patterns = [
        r"\+?\d[\d\s\-()]{8,}\d",
        r"\b\d{10}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def extract_linkedin(text):
    match = re.search(
        r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_/%]+",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(0)

    match = re.search(r"linkedin\.com/in/[A-Za-z0-9\-_/%]+", text, re.IGNORECASE)
    if match:
        return match.group(0)

    return None


def answer_from_text(question, text):
    q = question.lower().strip()

    if any(word in q for word in ["email", "mail", "mail id", "e-mail"]):
        email = extract_email(text)
        if email:
            return f"The email id in the document is {email}"

    if any(word in q for word in ["phone", "mobile", "contact number", "number"]):
        phone = extract_phone(text)
        if phone:
            return f"The phone number in the document is {phone}"

    if "linkedin" in q:
        linkedin = extract_linkedin(text)
        if linkedin:
            return f"The LinkedIn profile in the document is {linkedin}"

    chunks, index = create_rag_data(text)
    top_chunks = retrieve_top_chunks(question, chunks, index, k=8, candidate_k=20)
    return ask_llm(question, top_chunks)


if __name__ == "__main__":
    print("RAG system ready.")
    question = input("Ask a question: ")
    with open("data/output.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(answer_from_text(question, text))