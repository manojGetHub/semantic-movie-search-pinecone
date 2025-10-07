from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, INDEX_NAME, EMBED_MODEL_NAME

app = FastAPI(title="Semantic Movie Search")

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Embedding + Pinecone
model = SentenceTransformer(EMBED_MODEL_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
def search(q: str = Query(..., description="Natural language query"), k: int = 5):
    vec = model.encode([q], normalize_embeddings=True)[0].tolist()
    res = index.query(vector=vec, top_k=k, include_metadata=True)
    out = []
    for m in res["matches"]:
        md = m.get("metadata", {}) or {}
        out.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "title": md.get("title"),
            "genres": md.get("genres"),
            "year": md.get("year"),
            "overview": md.get("overview"),
        })
    return {"query": q, "results": out}
