from text_extraction import get_resume_text
from embedding import get_embedding
from faiss_index import compare_resume_with_jobs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# File paths
pdf_path = r"C:\Users\Omen\Downloads\resume.pdf"
docx_path = r"C:\Users\Omen\Downloads\resume.docx"
excel_path = r"C:\Users\Omen\Downloads\archive\Job.xlsx"

# Extract resume text
resume_text = get_resume_text(pdf_path, docx_path)

if resume_text:
    # Get the resume embedding
    resume_embedding = get_embedding(resume_text)

    # Get the most similar jobs using FAISS
    similar_jobs = compare_resume_with_jobs(resume_embedding, excel_path)

    # Print results
    print("\nMost Similar Jobs to Your Resume:")
    for job_title, job_description, similarity in similar_jobs:
        print(f"\nJob Title: {job_title}")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Job Description: {job_description[:500]}...")
        print("=" * 80)
else:
    print("Error: No resume text extracted.")
