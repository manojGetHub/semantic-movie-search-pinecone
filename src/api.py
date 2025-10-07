from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, INDEX_NAME, EMBED_MODEL_NAME

app = FastAPI(title="Semantic Movie Search API")
model = SentenceTransformer(EMBED_MODEL_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@app.get("/search")
def search(q: str = Query(..., description="Natural language query"), k: int = 5):
    vec = model.encode([q], normalize_embeddings=True)[0].tolist()
    res = index.query(vector=vec, top_k=k, include_metadata=True)
    out = []
    for m in res["matches"]:
        out.append({
            "id": m["id"],
            "score": m["score"],
            "title": m["metadata"]["title"],
            "genres": m["metadata"]["genres"],
            "year": m["metadata"]["year"],
            "overview": m["metadata"]["overview"]
        })
    return {"query": q, "results": out}
