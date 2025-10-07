import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, INDEX_NAME, EMBED_MODEL_NAME

# 1) Load model once (fast & local)
model = SentenceTransformer(EMBED_MODEL_NAME)

# 2) Load CSV
df = pd.read_csv("data/movies.csv")

# 3) Build text to embed (title + overview works well)
def text_for_row(r):
    return f"Title: {r['title']}. Overview: {r['overview']}. Genres: {r['genres']}. Year: {r['year']}"

texts = [text_for_row(r) for _, r in df.iterrows()]
ids = df["id"].tolist()

# 4) Compute embeddings
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

# 5) Prepare vectors with metadata
vectors = []
for i, vid in enumerate(ids):
    meta = {
        "title": df.loc[i, "title"],
        "overview": df.loc[i, "overview"],
        "genres": df.loc[i, "genres"],
        "year": int(df.loc[i, "year"])
    }
    vectors.append({"id": vid, "values": embeddings[i].tolist(), "metadata": meta})

# 6) Upsert to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
# Upsert accepts list[{"id","values","metadata"}]
print(f"Upserting {len(vectors)} vectors...")
index.upsert(vectors=vectors)
print("Ingest complete.")
