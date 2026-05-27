import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""

    # Open PDF
    doc = fitz.open(pdf_path)

    # Read every page
    for page in doc:
        text += page.get_text()

    return text


# Testing
if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    with open("data/output.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("Text extracted successfully!")