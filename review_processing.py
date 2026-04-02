import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("🔄 Loading translated dataset...")

# CHANGE THIS TO YOUR TRANSLATED FILE
input_path = r"C:\laragon\www\web\lokalkita\model\clean_data\review\review_cleaned_translated.csv"

# Final output file
output_path = r"C:\laragon\www\web\lokalkita\model\clean_data\review\review_clean_final.csv"

df = pd.read_csv(input_path, encoding="utf-8")
print("✅ Loaded! Rows:", len(df))


# ============================================================
# STEP 4: REMOVE ENGLISH STOPWORDS
# ============================================================

print("\n🔄 Step 4: Removing English stopwords...")

# Download once
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    if not isinstance(text, str):
        return ""  # handle NaN

    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered)

df["clean_no_stopwords"] = df["translated_review"].apply(remove_stopwords)

print("✅ Stopwords removed!")


# ============================================================
# STEP 5: FINAL CLEANUP
# ============================================================

print("\n🔄 Step 5: Final text cleanup...")

def final_clean(text):
    if not isinstance(text, str):
        return ""   # convert NaN / floats to empty string

    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip()


df["clean_no_stopwords"] = df["clean_no_stopwords"].apply(final_clean)

print("✨ Final cleaning done!")


# ============================================================
# STEP 6: SENTIMENT POLARITY SCORING
# ============================================================

print("\n🔄 Step 6: Sentiment scoring...")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if isinstance(text, str):
        score = analyzer.polarity_scores(text)
        return score["compound"]   # -1 to +1
    return 0

df["sentiment_score"] = df["translated_review"].apply(get_sentiment)

print("✅ Sentiment scoring complete!")
print(df[["translated_review", "sentiment_score"]].head())


# ============================================================
# STEP 7: SAVE FINAL CLEANED FILE
# ============================================================

df.to_csv(output_path, index=False, encoding="utf-8")

print("\n🎉 ALL DONE!")
print("Final cleaned dataset saved to:")
print(output_path)
