import psycopg2

# PostgreSQL connection details
DB_HOST = "localhost"
DB_NAME = "job_data"
DB_USER = "postgres"
DB_PASSWORD = "Eenadu@1"
DB_PORT = "5433"

def get_db_connection():
    """Creates and returns a PostgreSQL database connection and cursor."""
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )
    cursor = conn.cursor()
    return conn, cursor
