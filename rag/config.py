import os
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("MONGODB_DB", "rag_db")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "email_chunks")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp")