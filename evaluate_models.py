import numpy as np
import pandas as pd

from hybrid_scoring import search_recommendation
from personalized_model import personalized_recommendation, build_user_profile
from hybrid_scoring import df_meta


# ============================
# METRICS
# ============================

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_k) & relevant_set) / k


def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0
    return len(set(recommended_k) & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(recommended, relevant):
    relevant_set = set(relevant)
    for idx, item in enumerate(recommended, start=1):
        if item in relevant_set:
            return 1 / idx
    return 0


def hit_rate_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return int(len(set(recommended_k) & relevant_set) > 0)

# ============================
# TEST USERS (SIMULATED)
# ============================

test_users = [
    {
        "user_id": 1,
        "liked": ["EX12", "EX25"],
        "saved": ["EX30"],
        "interests": ["culture", "heritage"]
    },
    {
        "user_id": 2,
        "liked": ["EX5"],
        "saved": ["EX18", "EX22"],
        "interests": ["nature", "eco"]
    },
    {
        "user_id": 3,
        "liked": [],
        "saved": ["EX7"],
        "interests": ["food"]
    }
]


# ============================
# EVALUATION LOOP
# ============================

results = []

for user in test_users:
    user_id = user["user_id"]

    # ---------- Ground Truth ----------
    
    relevant_items = set(user["liked"] + user["saved"])

    # include interest-based relevance
    interest_mask = df_meta["category"].fillna("").str.lower().apply(
        lambda x: any(i in x for i in user["interests"])
    )
    relevant_items.update(df_meta[interest_mask]["item_id"].tolist())

    relevant_items = list(relevant_items)

    # ---------- HYBRID SEARCH ----------
    query = " ".join(user["interests"]) or "experience"
    hybrid_df = search_recommendation(query, top_k=10)
    hybrid_recs = hybrid_df["item_id"].tolist()

    # ---------- PERSONALIZED MODEL ----------
    user_vector = build_user_profile(
        user_key=user_id,
        liked_ids=user["liked"],
        saved_ids=user["saved"],
        interest_categories=user["interests"]
    )

    if user_vector is not None:
        personal_df = personalized_recommendation(
            user_key=user_id,
            user_vector=user_vector,
            liked_ids=user["liked"],
            saved_ids=user["saved"],
            top_k=10
        )
        personal_recs = personal_df["item_id"].tolist()
    else:
        personal_recs = []

    # ---------- METRICS ----------
    for model_name, recs in [
    ("Hybrid", hybrid_recs),
    ("Hybrid-Personalized", personal_recs)
]:

        results.append({
            "user_id": user_id,
            "model": model_name,
            "precision@10": precision_at_k(recs, relevant_items, 10),
            "recall@10": recall_at_k(recs, relevant_items, 10),
            "MRR": mean_reciprocal_rank(recs, relevant_items),
            "accuracy@10": hit_rate_at_k(recs, relevant_items, 10)
        })


# ============================
# RESULTS
# ============================

df_results = pd.DataFrame(results)

print("\n=== Evaluation Results ===")
print(df_results)

print("\n=== Average Scores ===")
print(df_results.groupby("model")[["precision@10", "recall@10", "MRR", "accuracy@10"]].mean())
