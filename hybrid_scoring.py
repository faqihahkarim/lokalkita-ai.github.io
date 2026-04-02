import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ======================================================
# CONFIG: DYNAMIC PATHS (FIXED FOR RENDER)
# ======================================================

# This finds the folder where hybrid_scoring.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to this file's location
metadata_path = os.path.join(base_dir, 'clean_data', 'metadata', 'metadata_cleaned.csv')
review_score_path = os.path.join(base_dir, 'review_model', 'review_weighted_scores.csv')
embeddings_path = os.path.join(base_dir, 'metadata_model', 'metadata_sbert_embeddings.npy')

# ======================================================
# LOAD DATA (ONCE AT STARTUP)
# ======================================================

print("🔄 Loading metadata & review scores...")

# Verify files exist before loading to avoid crashes
for p in [metadata_path, review_score_path, embeddings_path]:
    if not os.path.exists(p):
        print(f"❌ MISSING FILE: {p}")

df_meta = pd.read_csv(metadata_path)
df_review = pd.read_csv(review_score_path)

if "combined_text" not in df_meta.columns:
    raise ValueError("❌ 'combined_text' column missing. Run metadata_cleaning.py first.")

print("✅ Metadata rows:", len(df_meta))
print("✅ Review rows:", len(df_review))

# ======================================================
# TF-IDF (FIT ONCE)
# ======================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(
    df_meta["combined_text"].fillna("")
)

# ======================================================
# SBERT (LOAD ONCE)
# ======================================================

# Note: The first time this runs on Render, it will download the model.
# This may take a minute or two.
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = np.load(embeddings_path)

# ======================================================
# HELPER: NORMALIZATION
# ======================================================

def normalize_array(arr: np.ndarray) -> np.ndarray:
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

# ======================================================
# CORE HYBRID SCORING FUNCTION
# ======================================================

def compute_hybrid_scores(
    user_query: str,
    w_tfidf: float = 0.25,
    w_sbert: float = 0.55,
    w_review: float = 0.20
) -> pd.DataFrame:
    """
    Compute TF-IDF, SBERT, Review and Hybrid scores
    for a given user query.
    """

    # ---------- TF-IDF ----------
    user_vec = vectorizer.transform([user_query])
    tfidf_cos = cosine_similarity(user_vec, tfidf_matrix).flatten()
    tfidf_norm = normalize_array(tfidf_cos)

    # ---------- SBERT ----------
    query_emb = model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    sbert_cos = np.dot(embeddings, query_emb)
    sbert_norm = normalize_array(sbert_cos)

    # ---------- BASE DF ----------
    df = df_meta.copy()
    df["tfidf_norm"] = tfidf_norm
    df["sbert_norm"] = sbert_norm

    # ---------- REVIEW SCORE ----------
    df = df.merge(
        df_review[["item_id", "final_review_score"]],
        on="item_id",
        how="left"
    )

    df["final_review_score"] = df["final_review_score"].fillna(0.0)
    df["review_norm"] = normalize_array(
        df["final_review_score"].to_numpy()
    )

    # ---------- HYBRID SCORE ----------
    df["hybrid_score"] = (
        w_tfidf * df["tfidf_norm"] +
        w_sbert * df["sbert_norm"] +
        w_review * df["review_norm"]
    )

    return df.sort_values("hybrid_score", ascending=False)

# ======================================================
# PUBLIC SEARCH FUNCTION (FOR SYSTEM / API)
# ======================================================

def search_recommendation(
    query: str,
    top_k: int = 10,
    category: str | None = None
) -> pd.DataFrame:
    """
    Search recommendation with score breakdown.
    """

    ranked_df = compute_hybrid_scores(query)

    # Optional category filter
    if category is not None and "category" in ranked_df.columns:
        ranked_df = ranked_df[
            ranked_df["category"].str.lower() == category.lower()
        ]

    output_cols = [
        "item_id",
        "title",
        "category",
        "tfidf_norm",
        "sbert_norm",
        "review_norm",
        "hybrid_score"
    ]

    return ranked_df[output_cols].head(top_k)

# ======================================================
# DEMO / TEST
# ======================================================

if __name__ == "__main__":
    print("\n🔍 DEMO SEARCH: 'cooking class and food hunting'\n")

    results = search_recommendation(
        query="cooking class and food hunting",
        top_k=5,
    )

    print(results)