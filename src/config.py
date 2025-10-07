import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "movies-semantic-idx")

# Embedding model (local, free)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
EMBED_DIM = 384
