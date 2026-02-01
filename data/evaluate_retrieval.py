import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(MODELS_DIR, "movies_metadata.csv")
FAISS_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")

def evaluate():
    movies = pd.read_csv(DATA_PATH)
    index = faiss.read_index(FAISS_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    test_queries = [
        "pokemon animated fantasy adventure",
        "fun comedy movie with friends",
        "dark psychological thriller",
        "romantic movie with tragic ending",
        "superhero action movie"
    ]

    print("\n=== RETRIEVAL EVALUATION ===\n")

    for q in test_queries:
        q_vec = model.encode([q], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(q_vec, 5)

        print(f"\nQuery: {q}")
        for rank, idx in enumerate(indices[0], start=1):
            title = movies.iloc[idx]["title"]
            genres = movies.iloc[idx]["genres_text"]
            print(f"{rank}. {title} | {genres}")

if __name__ == "__main__":
    evaluate()
