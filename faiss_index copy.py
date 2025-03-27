import numpy as np
import faiss
import io
import pickle
from db_connection import get_db_connection
from embedding import get_long_text_embedding

# Function to serialize FAISS index
def serialize_faiss_index(index):
    buffer = io.BytesIO()
    faiss.write_index(index, buffer)
    return buffer.getvalue()

# Function to build FAISS index from existing job_postings table
def build_faiss_index_from_job_postings():
    # Get database connection
    conn, cursor = get_db_connection()
    
    try:
        # Create table for FAISS indices if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faiss_indices (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE,
                index_data BYTEA,
                dimension INTEGER,
                num_vectors INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Fetch job IDs and embeddings from existing job_postings table
        cursor.execute("SELECT job_id, embedding FROM job_postings WHERE embedding IS NOT NULL;")
        job_data = cursor.fetchall()
        
        if not job_data:
            print("No embeddings found in job_postings table.")
            return
        
        print(f"Found {len(job_data)} job embeddings in database.")
        
        # Extract job IDs and embeddings
        job_ids = []
        embeddings = []
        
        for job_id, embedding_data in job_data:
            job_ids.append(job_id)
            embeddings.append(np.array(embedding_data))
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Get embedding dimension
        d = embeddings_array.shape[1]
        
        # Create FAISS index
        quantizer = faiss.IndexFlatL2(d)
        
        # Calculate appropriate number of clusters
        num_vectors = len(embeddings_array)
        num_clusters = min(256, max(4, int(num_vectors / 39)))  # At least 39 vectors per cluster
        
        # Use IVFPQ for better performance
        index = faiss.IndexIVFPQ(quantizer, d, num_clusters, m=8, nbits=8)
        
        # Train and add vectors
        print("Training FAISS index...")
        index.train(embeddings_array)
        
        print("Adding vectors to index...")
        index.add(embeddings_array)
        
        # Set nprobe for search
        index.nprobe = 8
        
        # Serialize the index
        print("Serializing index...")
        serialized_index = serialize_faiss_index(index)
        
        # Store job IDs mapping in a separate table for reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faiss_job_mapping (
                faiss_index_name VARCHAR(255),
                vector_position INTEGER,
                job_id INTEGER,
                PRIMARY KEY (faiss_index_name, vector_position)
            );
        """)
        
        # Clear previous mappings
        cursor.execute("DELETE FROM faiss_job_mapping WHERE faiss_index_name = 'job_matching_index';")
        
        # Insert new mappings
        print("Storing job ID mappings...")
        for pos, job_id in enumerate(job_ids):
            cursor.execute(
                "INSERT INTO faiss_job_mapping (faiss_index_name, vector_position, job_id) VALUES (%s, %s, %s);",
                ('job_matching_index', pos, job_id)
            )
        
        # Store index in database
        print("Storing FAISS index in database...")
        cursor.execute("""
            INSERT INTO faiss_indices (name, index_data, dimension, num_vectors)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) 
            DO UPDATE SET 
                index_data = EXCLUDED.index_data,
                dimension = EXCLUDED.dimension,
                num_vectors = EXCLUDED.num_vectors,
                created_at = CURRENT_TIMESTAMP;
        """, ('job_matching_index', serialized_index, d, num_vectors))
        
        conn.commit()
        print(f"Successfully built and saved FAISS index with {num_vectors} job embeddings.")
    except Exception as e:
        conn.rollback()
        print(f"Error building FAISS index: {e}")
    finally:
        cursor.close()
        conn.close()

# Function to match resume with jobs using the stored FAISS index
def match_resume_with_jobs(resume_text, top_k=10):
    # Get resume embedding
    resume_embedding = get_long_text_embedding(resume_text) 
    
    # Get database connection
    conn, cursor = get_db_connection()
    
    try:
        # Load the FAISS index
        cursor.execute("SELECT index_data FROM faiss_indices WHERE name = 'job_matching_index';")
        result = cursor.fetchone()
        
        if not result:
            print("No FAISS index found in database.")
            return []
        
        # Deserialize the index
        buffer = io.BytesIO(result[0])
        index = faiss.read_index(buffer)
        
        # Convert resume embedding to numpy array
        resume_embedding_np = np.array(resume_embedding).astype('float32').reshape(1, -1)
        
        # Search for similar job embeddings
        distances, indices = index.search(resume_embedding_np, k=top_k)
        
        # Get job IDs for the matches
        matched_jobs = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            # Get job ID from mapping table
            cursor.execute(
                "SELECT job_id FROM faiss_job_mapping WHERE faiss_index_name = %s AND vector_position = %s;",
                ('job_matching_index', int(idx))
            )
            mapping_result = cursor.fetchone()
            
            if not mapping_result:
                continue
                
            job_id = mapping_result[0]
            
            # Get job details
            cursor.execute(
                "SELECT job_id, title, company, description FROM job_postings WHERE job_id = %s;",
                (job_id,)
            )
            job_result = cursor.fetchone()
            
            if job_result:
                job_id, title, company, description = job_result
                similarity_score = 1.0 - min(distances[0][i], 100) / 100  # Convert distance to similarity
                
                matched_jobs.append({
                    'job_id': job_id,
                    'title': title,
                    'company': company,
                    'similarity_score': float(similarity_score),
                    'description': description[:300] + '...' if len(description) > 300 else description
                })
        
        return matched_jobs
    finally:
        cursor.close()
        conn.close()

