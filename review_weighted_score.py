import pandas as pd
import numpy as np
import math

print("🔄 Loading cleaned review file...")

# CHANGE THIS to your cleaned final review file
input_path = r"C:\laragon\www\web\lokalkita\model\clean_data\review\review_clean_final.csv"

# Output file name
output_path = r"C:\laragon\www\web\lokalkita\model\review_model\review_weighted_scores.csv"

df = pd.read_csv(input_path, encoding="utf-8")
print("✅ Loaded! Rows:", len(df))


# ============================================================
# STEP 1: GROUP BY ITEM AND COMPUTE AVERAGE SENTIMENT
# ============================================================

print("\n🔄 Step 1: Calculating average sentiment per item...")

grouped = df.groupby("item_id")["sentiment_score"].agg(
    avg_sentiment="mean",
    review_count="count"
).reset_index()

print("✅ Average sentiment calculated!")
print(grouped.head())


# ============================================================
# STEP 2: COMPUTE WEIGHT = log(1 + review_count)
# ============================================================

print("\n Step 2: Computing weight using log(1 + review_count)...")

grouped["weight"] = grouped["review_count"].apply(lambda x: math.log(1 + x))

print("Weighting complete!")
print(grouped.head())


# ============================================================
# STEP 3: COMPUTE FINAL WEIGHTED SENTIMENT SCORE
# ============================================================

print("\n Step 3: Computing final weighted sentiment score...")

grouped["final_review_score"] = grouped["avg_sentiment"] * grouped["weight"]

print(" Final weighted sentiment score calculated!")
print(grouped.head())


# ============================================================
# STEP 4: SAVE OUTPUT
# ============================================================

grouped.to_csv(output_path, index=False, encoding="utf-8")

print("\n🎉 ALL DONE!")
print("Final weighted review score saved to:")
print(output_path)
