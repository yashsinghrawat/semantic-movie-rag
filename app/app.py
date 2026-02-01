import streamlit as st
import pandas as pd
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# MUST be first Streamlit call
st.set_page_config(
    page_title="AI Movie Recommendation Assistant",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f2027, #000000);
}
.movie-card {
    background: linear-gradient(145deg, #111827, #020617);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}
.movie-title {
    font-size: 22px;
    font-weight: 700;
}
.movie-genres {
    font-size: 13px;
    opacity: 0.7;
}
.similarity-bar {
    height: 8px;
    border-radius: 10px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
}
.intent-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    font-size: 12px;
    margin-right: 6px;
}
.footer {
    opacity: 0.4;
    text-align: center;
    margin-top: 40px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ================= CONFIG =================
st.set_page_config(page_title="AI Movie Assistant", layout="centered")
st.title("🎬 AI Movie Recommendation Assistant")
st.caption("Intent-Aware Semantic RAG")

SYSTEM_PROMPT = """
You are a movie recommendation assistant.

RULES:
- You may ONLY recommend movies that appear in the provided context.
- You must NOT invent or hallucinate movies.
- You must NOT use outside knowledge.

If the context is weak or imperfect:
- Explain that the match is approximate
- Still try to help using the closest available movies
- Ask clarifying questions if useful

Always explain WHY each recommended movie matches the user's request.
Be friendly, conversational, and helpful.
"""


INTENT_PROMPT = """
Extract structured intent from the user query.

Return ONLY valid JSON with these keys:
- mood
- audience
- preferred_genres (list)
- avoid (list)
- tone

If a field is not mentioned, use null or empty list.

User query:
{query}
"""

LENIENT_FALLBACK_MESSAGE = """
I couldn’t find a perfect match for what you asked, but I can still help 🙂

Here are some nearby or loosely related movies from my collection.
If you want better results, try telling me:
- the mood (fun, dark, chill, emotional)
- the type (animated, action, thriller, etc.)
- who you’re watching with

You can also say things like:
“something fun but not childish”
or
“surprise me with a chill movie”
"""


# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MOVIES_PATH = os.path.join(MODELS_DIR, "movies_metadata.csv")
FAISS_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")

# ================= LOADERS =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    return faiss.read_index(FAISS_PATH)

@st.cache_data
def load_movies():
    return pd.read_csv(MOVIES_PATH)

@st.cache_resource
def load_llm():
    return Groq()

# ================= LOAD =================
with st.spinner("Loading AI system..."):
    embedder = load_embedder()
    index = load_index()
    movies = load_movies()
    llm = load_llm()

st.success(f"Ready — {len(movies)} movies indexed")

# ================= HELPERS =================
def normalize_similarity(distances):
    max_d = distances.max()
    min_d = distances.min()
    return 1 - (distances - min_d) / (max_d - min_d + 1e-9)

def extract_intent(llm, query):
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You extract intent. Output JSON only."},
            {"role": "user", "content": INTENT_PROMPT.format(query=query)}
        ],
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "mood": None,
            "audience": None,
            "preferred_genres": [],
            "avoid": [],
            "tone": None
        }

def build_context(df):
    blocks = []
    for _, row in df.iterrows():
        blocks.append(
            f"""
Movie:
Title: {row['title']}
Genres: {row['genres_text']}
Similarity score: {row['similarity']:.2f}
Overview: {row['overview']}
"""
        )
    return "\n---\n".join(blocks)

def extract_allowed_titles(retrieved_df):
    return set(
        title.lower().strip()
        for title in retrieved_df["title"].tolist()
    )


def validate_llm_output(answer, allowed_titles):
    answer_lower = answer.lower()
    for word in answer_lower.split():
        if word.istitle():  # heuristic
            if word.lower() not in allowed_titles:
                return False
    return True


# ================= UI =================
query = st.text_input(
    "Describe the movie you want",
    placeholder="e.g. a fun animated movie for friends, not too childish"
)

if query:
    # -------- Retrieval --------
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, 15)

    sims = normalize_similarity(distances[0])
    retrieved = movies.iloc[indices[0]].copy()
    retrieved["similarity"] = sims

    #if retrieved["similarity"].max() < 0.25:
     #   st.warning("I couldn't find strong matches for your request.")
      #  st.stop()
    max_sim = retrieved["similarity"].max()

    LOW_CONF = 0.25
    WEAK_CONF = 0.40

    # -------- Intent Extraction --------

    def sanitize_intent(intent):
            return {
                "mood": intent.get("mood"),
                "audience": intent.get("audience"),
                "tone": intent.get("tone"),
                "preferred_genres": intent.get("preferred_genres") or [],
                "avoid": intent.get("avoid") or []
            }
    
    intent = sanitize_intent(extract_intent(llm,query))

    # -------- Intent-Aware Re-ranking --------

    intent_strength  = sum(

        1 for v in intent.values()
        if v not in (None, [],"" )
    )

    intent_weight = 0.05 * min(intent_strength, 3)

    def intent_adjust(row):
        score = row["similarity"]
        score += intent_weight
        genres = str(row["genres_text"]).lower()

        for g in intent["preferred_genres" ]:
            if g.lower() in genres:
                score += 0.05  # small boost

        for a in intent["avoid"]:
            if a.lower() in genres:
                score -= 0.05  # small penalty

        return max(0.0,min(score,1.0))

    retrieved["final_score"] = retrieved.apply(intent_adjust, axis=1)

    retrieved = retrieved.sort_values(
        by="final_score",
        ascending=False
    ).head(5)

    # -------- RAG Prompt --------
    context = build_context(retrieved)

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""

Context:
{context}

User request:
{query}
"""}
        ],
        temperature=0.4
    )

    # ================= OUTPUT =================
    st.markdown(" Detected Intent")

    for k, v in intent.items():
        if v:
            if isinstance(v, list):
                for item in v:
                    st.markdown(f"<span class='intent-chip'>{item}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='intent-chip'>{v}</span>", unsafe_allow_html=True)

    st.subheader("🤖 AI Recommendation")

    if max_sim < LOW_CONF:
        # Very weak match → conversational fallback
        st.info(LENIENT_FALLBACK_MESSAGE)

    elif max_sim < WEAK_CONF:
        # Weak match → still answer, but with caveat
        st.write(
            "I found some *loosely related* movies based on your request. "
            "They may not be a perfect match, but here’s why they could still work:"
        )
        st.write(response.choices[0].message.content)

    else:
        # Normal confident response
        st.write(response.choices[0].message.content)

    st.divider()
    st.subheader("🔍 Retrieved Movies")

    for _, row in retrieved.iterrows():
        score = min(row["final_score"], 1.0)

        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">🎞️ {row['title']}</div>
            <div class="movie-genres">{row['genres_text']}</div>
            <div style="margin: 8px 0;">
                <div class="similarity-bar" style="width:{int(score*100)}%"></div>
            </div>
            <small>Relevance: {score:.2f}</small>
            <p style="margin-top:10px;">{row['overview']}</p>
        </div>
        """, unsafe_allow_html=True)

    
    
st.markdown("""
<div class="footer">
Semantic RAG Movie Engine · Built with FAISS + Sentence Transformers + LLaMA-3
</div>
""", unsafe_allow_html=True)


