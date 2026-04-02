from fastapi import FastAPI, Query
import numpy as np

from hybrid_scoring import (
    search_recommendation,
    df_meta,
    embeddings,
    normalize_array
)

from personalized_model import (
    build_user_profile,
    personalized_recommendation,
    clear_user_cache
)

app = FastAPI(title="LokalKita Recommendation API")

# =====================================================
# SEARCH (HYBRID)
# =====================================================

@app.get("/search")
def search(query: str = Query(..., min_length=3)):
    results = search_recommendation(query)
    return results.to_dict(orient="records")


# =====================================================
# PERSONALIZED RECOMMENDATION (PRODUCTION)
# =====================================================

@app.get("/recommend/personalized")
def recommend_personalized(
    user_id: int,
    liked: list[int] = Query(default=[]),
    saved: list[int] = Query(default=[]),
    interests: list[str] = Query(default=[])
):
    user_vector = build_user_profile(
        user_key=user_id,
        liked_ids=liked,
        saved_ids=saved,
        interest_categories=interests
    )

    if user_vector is None:
        return []

    df = personalized_recommendation(
        user_key=user_id,
        user_vector=user_vector,
        liked_ids=liked,
        saved_ids=saved
    )

    # CONVERT EXxxx → INTEGER exp_id
    df["item_id"] = (
        df["item_id"]
        .astype(str)
        .str.replace("EX", "", regex=False)
        .astype(int)
    )

    return df.to_dict(orient="records")

#
# ===========
# DEBUG PERSONALIZATION (FOR EVALUATION / REPORT)
# =====================================================

@app.get("/debug/personalized")
def debug_personalized(
    liked: list[int] = Query(default=[]),
    saved: list[int] = Query(default=[]),
    interests: list[str] = Query(default=[])
):
    liked_ids = [f"EX{x}" for x in liked]
    saved_ids = [f"EX{x}" for x in saved]

    user_vector = build_user_profile(
        user_key="__debug__",
        liked_ids=liked_ids,
        saved_ids=saved_ids,
        interest_categories=interests
    )

    if user_vector is None:
        return {"message": "No user vector generated"}

    scores = np.dot(embeddings, user_vector)
    scores_norm = normalize_array(scores)

    df = df_meta[["item_id", "title"]].copy()
    df["personalized_score"] = scores_norm

    return (
        df.sort_values("personalized_score", ascending=False)
          .head(20)
          .to_dict(orient="records")
    )


# =====================================================
# CACHE CLEAR (CALLED FROM PHP)
# =====================================================

@app.get("/cache/clear")
def clear_cache(user_id: int):
    clear_user_cache(user_id)
    return {"status": "cache cleared", "user_id": user_id}
