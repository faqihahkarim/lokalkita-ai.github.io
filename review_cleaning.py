import pandas as pd
import emoji
import re
import time
from deep_translator import GoogleTranslator
import os

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("🔄 Loading dataset...")

input_path = r"C:\laragon\www\web\lokalkita\model\raw_data\raw_review.xlsx"
checkpoint_path = r"C:\laragon\www\web\lokalkita\model\clean_data\review_clean_checkpoint.csv"

df = pd.read_excel(input_path)
print(f"✅ Dataset loaded! Total rows: {len(df)}")

# ============================================================
# STEP 2: BASIC CLEANING (emoji, symbols)
# ============================================================
print("\n🔄 Step 2: Cleaning emojis & special characters...")

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def basic_clean(text):
    if isinstance(text, str):
        text = remove_emoji(text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^A-Za-z0-9\s,.!?]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return text

df["clean_review"] = df["review"].apply(basic_clean)
print("✅ Step 2 complete!")

# ============================================================
# STEP 3: TRANSLATION WITH SAFE CHECKPOINT (STARTS ONLY AFTER CREATED)
# ============================================================

print("\n Step 3: Translating reviews with checkpoint system")

checkpoint_path = r"C:\laragon\www\web\lokalkita\data\latest(231125)\review_clean_checkpoint.csv"

# If checkpoint exists -> resume
if os.path.exists(checkpoint_path):
    print("Checkpoint found! Resuming translation...")
    df_checkpoint = pd.read_csv(checkpoint_path, encoding='latin1')

    # If translated_review column exists → use it
    if "translated_review" in df_checkpoint.columns:
        df["translated_review"] = df_checkpoint["translated_review"]
    else:
        print("Checkpoint missing translated_review column. Starting fresh.")
        df["translated_review"] = ""

else:
    # No checkpoint file -> start fresh
    print("No checkpoint found. Starting translation from zero.")
    df["translated_review"] = ""

translator = GoogleTranslator(source='auto', target='en')
total = len(df)

for i in range(total):

    # Skip already translated rows
    if isinstance(df.loc[i, "translated_review"], str) and len(df.loc[i, "translated_review"]) > 1:
        continue

    text = df.loc[i, "clean_review"]

    # Try translation
    try:
        translated = translator.translate(text)
        df.loc[i, "translated_review"] = translated
    except Exception as e:
        print(f"Translation error at row {i}: {e}")
        df.loc[i, "translated_review"] = text  # fallback

    # Progress print every 100 rows
    if i % 100 == 0:
        percent = round((i / total) * 100, 2)
        print(f"Progress: {i}/{total} rows ({percent}%)")

    # Save checkpoint every 500 rows
    if i % 500 == 0:
        df.to_csv(checkpoint_path, index=False, encoding='utf-8')
        print(f"Checkpoint saved at row {i}")

# Final save
df.to_csv(checkpoint_path, index=False, encoding='utf-8')
print("Translation complete! Final checkpoint saved.")


