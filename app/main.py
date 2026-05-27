import os
from fastapi import FastAPI, UploadFile, File, Query, HTTPException

from app.pdf_utils import extract_text_from_pdf
from app.rag_bot import create_rag_data, answer_from_text, split_text

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_text = None
current_chunks = None
current_index = None
current_filename = None


@app.get("/")
def home():
    return {"message": "RAG API is running successfully!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_text, current_chunks, current_index, current_filename

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF.")

    current_text = text
    current_chunks, current_index = create_rag_data(text)
    current_filename = file.filename

    with open("data/output.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "filename": file.filename,
        "message": "PDF uploaded and index built successfully!",
        "preview": text[:500]
    }


@app.get("/ask")
def ask(question: str = Query(..., description="Your question about the uploaded document")):
    global current_text, current_chunks, current_index, current_filename

    if current_text is None or current_chunks is None or current_index is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    answer = answer_from_text(question, current_text)

    # Also return top source chunks for UI display
    top_chunks = []
    q = question.lower().strip()

    if current_chunks and current_index:
        from app.rag_bot import retrieve_top_chunks
        top_chunks = retrieve_top_chunks(question, current_chunks, current_index, k=4, candidate_k=12)

    return {
        "filename": current_filename,
        "question": question,
        "answer": answer,
        "sources": top_chunks
    }