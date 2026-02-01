# 🎬 AI Movie Recommendation Assistant  
### Intent-Aware Semantic Search + RAG System (FAISS × LLaMA-3)

🚀 **Live Demo:** https://<your-streamlit-app-url>

---

## 🔥 What This Project Is (and Is NOT)

❌ Not a keyword search  
❌ Not a traditional recommender system  
❌ Not a ChatGPT wrapper  

✅ **A production-style, intent-aware semantic retrieval system**  
✅ **Grounded Retrieval-Augmented Generation (RAG)**  
✅ **User-controlled, explainable recommendations**

This system understands **meaning**, not keywords.

---

## 🧠 Example

**User input:**
> *“brainrot movies to watch with friends, not too childish”*

**System behavior:**
- Interprets *intent* (mood, audience, exclusions)
- Retrieves semantically similar movies using dense embeddings
- Re-ranks results using intent + user preferences
- Generates a **grounded explanation** using only retrieved data
- Falls back conversationally if the query is vague

---

## 🏗️ System Architecture

User Query
↓
SentenceTransformer (Dense Embeddings)
↓
FAISS Vector Search
↓
Intent Extraction (LLaMA-3)
↓
Intent-Aware Re-Ranking
↓
RAG Context Builder
↓
LLaMA-3 (Groq API)
↓
Grounded, Explainable Output


---

## ⚙️ Core Features

- 🔍 **Semantic Search** (FAISS + MiniLM embeddings)
- 🧠 **Intent Extraction** (mood, audience, preferred/avoided genres)
- 🎛️ **User Controls**
  - Genre include / exclude
  - Result count (Top-K)
  - Strictness slider (exploration vs precision)
- 🧩 **RAG (Retrieval-Augmented Generation)**
  - Model can ONLY use retrieved movies
  - Hallucination-safe by design
- 🗣️ **Lenient Conversational Fallback**
  - Responds helpfully even for vague queries
- 🎨 **Premium UI**
  - Movie cards, relevance bars, intent chips
- ☁️ **Deployed on Streamlit Cloud**

---

## 🛡️ Hallucination Safety (Important)

The LLM is **strictly constrained**:
- It cannot mention movies outside the retrieved context
- If data is insufficient, it explains limitations instead of guessing
- This ensures **trustworthy, grounded recommendations**

---

## 🧪 Tech Stack

| Component | Technology |
|---------|-----------|
| Embeddings | `sentence-transformers (all-MiniLM-L6-v2)` |
| Vector DB | `FAISS (CPU)` |
| LLM | `LLaMA-3.1-8B` via Groq |
| Backend | Python |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

semantic/
├── app/
│ └── app.py # Streamlit application
├── models/
│ ├── movies_metadata.csv # Movie dataset
│ └── faiss_index.bin # Vector index
├── requirements.txt
└── README.md

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
