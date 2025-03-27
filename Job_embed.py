import numpy as np
import torch
from db_connection import get_db_connection
from embedding import get_long_text_embedding 


# Get database connection and cursor
conn, cursor = get_db_connection()

# Fetch job descriptions
cursor.execute("SELECT job_id, description FROM job_postings WHERE description IS NOT NULL;")
jobs = cursor.fetchall()

# Process and store embeddings
for job_id, description in jobs:
    embedding = get_long_text_embedding(description)  # Compute embedding
    embedding = np.array(embedding).tolist()  # Convert to list

    # Store embedding in the database
    cursor.execute(
        "UPDATE job_postings SET embedding = %s WHERE job_id = %s;",
        (embedding, job_id)
    )

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print(f"Updated {len(jobs)} rows with embeddings.")