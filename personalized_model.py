import numpy as np
import pandas as pd
# Import the fixed variables from your other file
from hybrid_scoring import df_meta, embeddings, normalize_array

# =====================================================
# BUILD USER PROFILE VECTOR
# =====================================================

def build_user_profile(
    user_key,
    liked_ids=None,
    saved_ids=None,
    interest_categories=None,
    w_like=0.4,
    w_save=0.2,
    w_interest=0.4,
    max_interests=2,
    interest_sample_k=10,
    seed=42
):
    liked_ids = liked_ids or []
    saved_ids = saved_ids or []
    interest_categories = interest_categories or []

    vectors = []
    weights = []

    # ======================
    # LIKES
    # ======================
    if liked_ids:
        # Convert to strings if your item_ids in CSV have "EX" prefix
        idx = df_meta[df_meta["item_id"].isin(liked_ids)].index
        if len(idx) > 0:
            vectors.append(np.mean(embeddings[idx], axis=0))
            weights.append(w_like)

    # ======================
    # SAVES
    # ======================
    if saved_ids:
        idx = df_meta[df_meta["item_id"].isin(saved_ids)].index
        if len(idx) > 0:
            vectors.append(np.mean(embeddings[idx], axis=0))
            weights.append(w_save)

    # ======================
    # INTERESTS (CATEGORY OR TAGS)
    # ======================
    if interest_categories:
        interests = [
            str(c).lower().strip()
            for c in interest_categories
            if c and str(c).strip()
        ][:max_interests]

        cat_series = df_meta["category"].fillna("").str.lower()
        tag_series = df_meta["tags"].fillna("").str.lower()

        mask = cat_series.apply(
            lambda x: any(i in x for i in interests)
        ) | tag_series.apply(
            lambda x: any(i in x for i in interests)
        )

        idx = df_meta[mask].index

        if len(idx) > 0:
            rng = np.random.default_rng(seed)
            sampled_idx = rng.choice(
                idx.to_numpy(),
                size=min(interest_sample_k, len(idx)),
                replace=False
            )
            vectors.append(np.mean(embeddings[sampled_idx], axis=0))
            weights.append(w_interest)

    # ======================
    # FAIL SAFE
    # ======================
    if not vectors:
        return None

    # ======================
    # FINAL USER VECTOR
    # ======================
    return np.average(vectors, axis=0, weights=weights)


# =====================================================
# PERSONALIZED RECOMMENDATION
# =====================================================

def personalized_recommendation(
    user_key,
    user_vector,
    liked_ids=None,
    saved_ids=None,
    top_k=5,
    alpha=0.6  # hybrid weight
):
    # ---------- Personalized similarity ----------
    personal_scores = np.dot(embeddings, user_vector)
    personal_scores = normalize_array(personal_scores)

    df = pd.DataFrame({
        "item_id": df_meta["item_id"],
        "category": df_meta["category"],
        "personal_score": personal_scores
    })

    # ---------- Hybrid relevance (content + reviews) ----------
    # We import this INSIDE the function to avoid circular import errors on Render
    from hybrid_scoring import compute_hybrid_scores
    
    # "experience" is the default query to get base hybrid scores
    hybrid_df = compute_hybrid_scores("experience")
    df = df.merge(
        hybrid_df[["item_id", "hybrid_score"]],
        on="item_id",
        how="left"
    )

    df["hybrid_score"] = df["hybrid_score"].fillna(0.0)

    # ---------- Final fused score ----------
    df["final_score"] = (
        alpha * df["hybrid_score"] +
        (1 - alpha) * df["personal_score"]
    )

    # ---------- Exclude interacted ----------
    exclude = set((liked_ids or []) + (saved_ids or []))
    if exclude:
        df = df[~df["item_id"].isin(exclude)]

    # ---------- Rank + controlled diversity ----------
    top = df.sort_values("final_score", ascending=False).head(30)

    # Use a fixed seed based on user_key so recommendations are stable
    try:
        seed_val = int(user_key) if str(user_key).isdigit() else 42
    except:
        seed_val = 42

    result = top.sample(
        n=min(top_k, len(top)),
        random_state=seed_val
    )

    return result.reset_index(drop=True)


# =====================================================
# CACHE CLEAR
# =====================================================

def clear_user_cache(user_key):
    # placeholder (no caching used now)
    return