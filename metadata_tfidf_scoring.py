import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# ==========================
# CONFIG: PATHS
# ==========================

# Your cleaned metadata file
metadata_path = r"C:\laragon\www\web\lokalkita\model\clean_data\metadata\metadata_cleaned.csv"

# Output score file (optional)
output_path = r"C:\laragon\www\web\lokalkita\model\metadata_model\metadata_tfidf_scores.csv"

# Make sure folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)


# ==========================
# STEP 1 — Load metadata
# ==========================

print("🔄 Loading metadata...")
df = pd.read_csv(metadata_path)
print("✅ Rows loaded:", len(df))

if "combined_text" not in df.columns:
    raise ValueError("❌ 'combined_text' column missing. Run metadata_cleaning.py first.")


# ==========================
# STEP 2 — Build TF-IDF vectorizer
# ==========================

print("\n🔄 Building TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000  # prevents overfitting
)

tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

print("✅ TF-IDF matrix shape:", tfidf_matrix.shape)


# ==========================
# STEP 3 — Define function for scoring new input
# ==========================

def compute_tfidf_scores(user_query):
    """
    Convert the user's query into TF-IDF vector,
    compute cosine similarity,
    normalize scores, return sorted DataFrame.
    """
    print("\n Processing query:", user_query)

    # Vectorize user query (single row)
    user_vec = vectorizer.transform([user_query])

    # Cosine similarity to all items -> (1 × N)
    cosine_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    # Save raw score
    df["tfidf_raw_score"] = cosine_scores

    # Normalize to 0–1
    min_val = cosine_scores.min()
    max_val = cosine_scores.max()

    if max_val - min_val == 0:
        df["tfidf_normalized"] = 0
    else:
        df["tfidf_normalized"] = (cosine_scores - min_val) / (max_val - min_val)

    # Sort by highest score
    result = df.sort_values("tfidf_normalized", ascending=False)

    return result


# ==========================
# STEP 4 — Test with sample input
# ==========================

example_query = "cultural art gallery with modern exhibits"
print("\n Running example query:", example_query)

result_df = compute_tfidf_scores(example_query)

print("\nTOP 5 RESULTS:")
print(result_df[["item_id", "title", "tfidf_raw_score", "tfidf_normalized"]].head())


# ==========================
# STEP 5 — Save results
# ==========================

result_df[["item_id", "title", "tfidf_raw_score", "tfidf_normalized"]].to_csv(output_path, index=False)
print("\n🎉 TF-IDF scoring complete!")
print("Saved to:", output_path)
