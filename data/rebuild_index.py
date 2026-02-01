import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(MODELS_DIR, "movies_metadata.csv")
FAISS_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")

def rebuild_index():
    print("Loading dataset...")
    movies = pd.read_csv(DATA_PATH)

    texts = (
        movies["title"].fillna("") + ". " +
        movies["genres_text"].fillna("") + ". " +
        movies["overview"].fillna("")
    ).tolist()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(texts)} movies...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)

    print(f"FAISS index rebuilt with {index.ntotal} vectors")

if __name__ == "__main__":
    rebuild_index()