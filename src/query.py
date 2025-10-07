import sys
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, INDEX_NAME, EMBED_MODEL_NAME

query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "movie about survival on Mars"

# 1) Embed query
model = SentenceTransformer(EMBED_MODEL_NAME)
qvec = model.encode([query_text], normalize_embeddings=True)[0].tolist()

# 2) Query Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
res = index.query(vector=qvec, top_k=3, include_metadata=True)

# 3) Pretty print
print(f"\nQuery: {query_text}\nTop results:")
for match in res["matches"]:
    score = match["score"]
    meta = match["metadata"]
    print(f"- {meta['title']} (score: {score:.3f})")
    print(f"  Genres: {meta['genres']}  |  Year: {meta['year']}")
    print(f"  {meta['overview']}\n")
