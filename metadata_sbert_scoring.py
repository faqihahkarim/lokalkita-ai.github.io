import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

# ==========================
# CONFIG: PATHS
# ==========================

metadata_path = r"C:\laragon\www\web\lokalkita\model\clean_data\metadata\metadata_cleaned.csv"

# save SBERT embeddings
embeddings_path = r"C:\laragon\www\web\lokalkita\model\metadata_model\metadata_sbert_embeddings.npy"

# save scoring results
output_path = r"C:\laragon\www\web\lokalkita\model\metadata_model\metadata_sbert_scores.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ==========================
# STEP 1 — Load metadata
# ==========================

print("🔄 Loading metadata...")
df = pd.read_csv(metadata_path)
print("✅ Rows loaded:", len(df))

if "combined_text" not in df.columns:
    raise ValueError("❌ 'combined_text' column missing. Run metadata_cleaning.py first.")

texts = df["combined_text"].fillna("").tolist()

# ==========================
# STEP 2 — Load SBERT model
# ==========================

print("\n🔄 Loading SBERT model (this may take a while the first time)...")

# You can change model if you want; this one is small & good
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

print("✅ Model loaded:", model_name)

# ==========================
# STEP 3 — Encode metadata (or load cached)
# ==========================

if os.path.exists(embeddings_path):
    print("\n🔁 Found existing embeddings file. Loading from disk...")
    embeddings = np.load(embeddings_path)
    print("✅ Embeddings loaded. Shape:", embeddings.shape)
else:
    print("\n🔄 Encoding metadata combined_text with SBERT...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # cos-sim becomes dot product
    )
    print("✅ Embeddings created. Shape:", embeddings.shape)

    # save for reuse
    np.save(embeddings_path, embeddings)
    print("💾 Embeddings saved to:", embeddings_path)

# ==========================
# STEP 4 — Function to compute SBERT scores for a query
# ==========================

def compute_sbert_scores(user_query: str) -> pd.DataFrame:
    """
    Encode user query, compute cosine similarity with all items,
    normalize scores to 0–1 and return sorted DataFrame.
    """
    print("\n🔍 Processing SBERT query:", user_query)

    # encode query
    query_emb = model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]  # shape: (D,)

    # cosine similarity = dot product (because we normalized)
    cosine_scores = np.dot(embeddings, query_emb)  # shape: (N,)

    df["sbert_raw_score"] = cosine_scores

    # normalize 0–1
    min_val = cosine_scores.min()
    max_val = cosine_scores.max()

    if max_val - min_val == 0:
        df["sbert_normalized"] = 0.0
    else:
        df["sbert_normalized"] = (cosine_scores - min_val) / (max_val - min_val)

    result = df.sort_values("sbert_normalized", ascending=False)
    return result

# ==========================
# STEP 5 — Test with example queries
# ==========================

example_query_1 = " waterfall camping in the nature"
example_query_2 = "batik workshop"

print("\n⚡ Running example SBERT query 1:", example_query_1)
result1 = compute_sbert_scores(example_query_1)

print("\nSBERT TOP 5 for:", example_query_1)
print(result1[["item_id", "title", "sbert_raw_score", "sbert_normalized"]].head())

print("\n⚡ Running example SBERT query 2:", example_query_2)
result2 = compute_sbert_scores(example_query_2)

print("\nSBERT TOP 5 for:", example_query_2)
print(result2[["item_id", "title", "sbert_raw_score", "sbert_normalized"]].head())

# ==========================
# STEP 6 — Save last result (optional)
# ==========================

# Here I save the last result (example_query_2), but you can change this later
result2[["item_id", "title", "sbert_raw_score", "sbert_normalized"]].to_csv(
    output_path, index=False, encoding="utf-8"
)

print("\n🎉 SBERT scoring complete!")
print("Saved to:", output_path)
