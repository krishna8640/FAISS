import os
import pdfplumber
import docx2txt

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found!")
        return None
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip() if text else None

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    if not os.path.exists(docx_path):
        print(f"Error: File '{docx_path}' not found!")
        return None
    return docx2txt.process(docx_path).strip()

# Function to get resume text (prioritize PDF)
def get_resume_text(pdf_path, docx_path):
    return extract_text_from_pdf(pdf_path) or extract_text_from_docx(docx_path)
