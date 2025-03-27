import faiss
import numpy as np
import pandas as pd
from embedding import get_long_text_embedding

# Function to read job descriptions from an Excel file
def read_job_descriptions(excel_path):
    df = pd.read_excel(excel_path)
    return df['Job Title'].tolist(), df['Job Description'].tolist()

# Function to create FAISS index
def create_faiss_index(job_descriptions):
    job_embeddings = [get_long_text_embedding(desc) for desc in job_descriptions]
    job_embeddings = np.array(job_embeddings).astype('float32')

    num_clusters = 256  # Adjust based on dataset size
    d = job_embeddings.shape[1]  # Embedding dimension

    # Create an Inverted Multi-Index (IMI)
    quantizer = faiss.IndexFlatL2(d)  # Base index
    index = faiss.IndexIVFFlat(quantizer, d, num_clusters, faiss.METRIC_L2)

    # Train the index (required for IVFFlat)
    index.train(job_embeddings)  # Train with job embeddings
    index.add(job_embeddings)  # Add embeddings to the index

    return index, job_embeddings

# Function to compare resume with job descriptions using FAISS
def compare_resume_with_jobs(resume_embedding, excel_path):
    job_titles, job_descriptions = read_job_descriptions(excel_path)
    index, job_embeddings = create_faiss_index(job_descriptions)

    index.nprobe = 10  # Number of clusters to search (tune for speed vs. accuracy)

    resume_embedding_np = resume_embedding.astype('float32').reshape(1, -1)
    D, I = index.search(resume_embedding_np, k=100)  # Top 100 matches

    return [(job_titles[idx], job_descriptions[idx], D[0][i]) for i, idx in enumerate(I[0])]
