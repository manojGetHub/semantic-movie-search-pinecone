# Line-by-line with comments
from pinecone import Pinecone, ServerlessSpec
from src.config import PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION, INDEX_NAME, EMBED_DIM

# 1) Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2) If index exists, skip; else create serverless index
existing = [idx['name'] for idx in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,      # 'aws' or 'gcp'
            region=PINECONE_REGION     # e.g., 'us-east-1'
        ),
        metric="cosine"               # cosine for sentence embeddings
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

print("Done. (Note: serverless index becomes ready shortly.)")
