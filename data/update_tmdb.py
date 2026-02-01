import os
import requests
import pandas as pd
from datetime import datetime

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR,"models","movies_metadata.csv")

TMDB_URL = "https://api.themoviedb.org/3/movie/popular"

def fetch_latest_movies(pages = 3):
    movies = []

    for page in range(1,pages+1):
        response = requests.get(

            TMDB_URL,
            params={
                "api_key": TMDB_API_KEY,
                "language":"en-US",
                "page":page
            }
        )

        response.raise_for_status()
        data = response.json()['results']

        for m in data:
            movies.append({

                "tmdb_id":m['id'],
                'title':m['title'],
                'overview':m["overview"],
                'genres_text':"",
                'release_date':m.get('release_date','')

            })

    return pd.DataFrame(movies)

def  update_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("movies_metadata.csv not found")
    
    existing = pd.read_csv(DATA_PATH)

    latest = fetch_latest_movies()
    combined = pd.concat([existing, latest],ignore_index = True)

    combined = combined.drop_duplicates(subset =['title']).reset_index(drop  = True)
   
    combined.to_csv(DATA_PATH, index = False)

    print(f'Dataset updated : {len(existing)} --> {len(combined)} movies')

if __name__ == "__main__":
    update_dataset()
