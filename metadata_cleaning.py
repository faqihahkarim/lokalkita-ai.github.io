import pandas as pd
import re
import math
import os
import emoji

# ==========================
# CONFIG: INPUT & OUTPUT PATH
# ==========================

# raw file path
input_path = r"C:\laragon\www\web\lokalkita\model\raw_data\raw_metadata.xlsx"

output_path = r"C:\laragon\www\web\lokalkita\model\clean_data\metadata\metadata_cleaned.csv"

print("🔄 Loading raw metadata...")
df = pd.read_excel(input_path)
print("✅ Loaded! Rows:", len(df))
print(df.head())


# ==========================
# HELPER FUNCTIONS
# ==========================

def normalize_missing(value):
    """Convert dash-like / blank values into 'Not Mentioned'."""
    if pd.isna(value):
        return "Not Mentioned"
    if isinstance(value, str):
        text = value.strip()
        if text in ["-", "–", "—", ""]:
            return "Not Mentioned"
    return value


def clean_text_basic(text):
    """Remove emojis, normalize spaces; keep content."""
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_tags(text):
    """Lowercase tags, strip spaces, remove duplicates, normalize commas."""
    if not isinstance(text, str):
        return ""
    # split by comma
    parts = [p.strip().lower() for p in text.split(",")]
    # remove empties & duplicates (keep order)
    seen = set()
    cleaned = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            cleaned.append(p)
    return ", ".join(cleaned)


# --- Day expansion helpers ---

DAY_MAP = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "weds": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}

def expand_single_day(token):
    token = token.strip().lower()
    return DAY_MAP.get(token, token.capitalize())


def expand_day_range(value):
    """
    Expand things like:
    'Mon-Sat' -> 'Monday – Saturday'
    'Mon - Fri' -> 'Monday – Friday'
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "" or text.lower() in ["not mentioned", "-"]:
        return "Not Mentioned"

    low = text.lower()

    # Handle common phrases
    if "daily" in low:
        return "Daily"
    if "everyday" in low:
        return "Daily"
    if "by appointment" in low:
        return "Daily by appointment"
    if "upon booking" in low:
        return "Upon Booking"

    # Normalize dashes
    low = low.replace("—", "-").replace("–", "-")

    # If it's a range like mon-sat
    if "-" in low:
        parts = [p.strip() for p in low.split("-") if p.strip()]
        if len(parts) == 2:
            start = expand_single_day(parts[0])
            end = expand_single_day(parts[1])
            return f"{start} – {end}"

    # If single day abbreviation like 'Mon'
    return expand_single_day(low)


# --- Time standardization helpers ---

TIME_PATTERN = re.compile(
    r"(?P<hour>\d{1,2})([:.](?P<minute>\d{2}))?\s*(?P<ampm>(am|pm|a\.m\.|p\.m\.)?)",
    re.IGNORECASE,
)

def parse_time_token(token):
    """
    Parse a time token like '9:00 am', '8.00PM', '8 am'
    Return (hour, minute, ampm) or None.
    """
    token = token.strip()
    match = TIME_PATTERN.search(token)
    if not match:
        return None

    hour = int(match.group("hour"))
    minute = match.group("minute")
    minute = int(minute) if minute else 0
    ampm = match.group("ampm").lower().replace(".", "")

    if ampm == "":
        # If no am/pm given, we can't safely guess. Return None.
        return None

    # normalize to 'a.m.' / 'p.m.'
    if ampm.startswith("a"):
        ampm_norm = "a.m."
    else:
        ampm_norm = "p.m."

    return hour, minute, ampm_norm


def format_time(hour, minute, ampm_norm):
    return f"{hour:d}.{minute:02d} {ampm_norm}"


def standardize_hours(value):
    """
    Standardize operating hours to format:
    '8.00 a.m. – 8.00 p.m.'
    """
    if not isinstance(value, str):
        return "Not Mentioned"

    text = value.strip()
    if text == "" or text.lower() in ["not mentioned", "-", "none"]:
        return "Not Mentioned"

    low = text.lower()
    # Normalize dashes
    low = low.replace("—", "-").replace("–", "-").replace(" to ", "-")

    # Split into start and end by '-'
    parts = [p.strip() for p in low.split("-") if p.strip()]
    if len(parts) != 2:
        # Could not parse range properly, just clean spaces and return original
        return clean_text_basic(text)

    start_raw, end_raw = parts[0], parts[1]

    start_parsed = parse_time_token(start_raw)
    end_parsed = parse_time_token(end_raw)

    if not start_parsed or not end_parsed:
        # Fallback if parsing fails
        return clean_text_basic(text)

    sh, sm, sampm = start_parsed
    eh, em, eampm = end_parsed

    start_str = format_time(sh, sm, sampm)
    end_str = format_time(eh, em, eampm)

    return f"{start_str} – {end_str}"


# ==========================
# STEP 1: NORMALIZE MISSING VALUES
# ==========================

print("\n🔄 Step 1: Normalizing missing values (dashes -> 'Not Mentioned')...")

cols_to_normalize = [
    "description",
    "tags",
    "operating hours",
    "operating days",
    "closing days",
    "more_info",
    "contact",
]

for col in cols_to_normalize:
    if col in df.columns:
        df[col] = df[col].apply(normalize_missing)

print("✅ Step 1 done.")


# ==========================
# STEP 2: CLEAN DESCRIPTION & TAGS
# ==========================

print("\n🔄 Step 2: Cleaning description and tags...")

# Clean description text
df["clean_description"] = df["description"].apply(clean_text_basic)

# Clean tags
df["clean_tags"] = df["tags"].apply(clean_tags)

print("✅ Step 2 done.")
print(df[["description", "clean_description", "tags", "clean_tags"]].head())


# ==========================
# STEP 3: STANDARDIZE OPERATING HOURS & DAYS
# ==========================

print("\n🔄 Step 3: Standardizing operating hours and days...")

if "operating hours" in df.columns:
    df["clean_operating_hours"] = df["operating hours"].apply(standardize_hours)
else:
    df["clean_operating_hours"] = "Not Mentioned"

if "operating days" in df.columns:
    df["clean_operating_days"] = df["operating days"].apply(expand_day_range)
else:
    df["clean_operating_days"] = "Not Mentioned"

if "closing days" in df.columns:
    df["clean_closing_days"] = df["closing days"].apply(expand_day_range)
else:
    df["clean_closing_days"] = "Not Mentioned"

print("✅ Step 3 done.")
print(df[["operating hours", "clean_operating_hours", "operating days", "clean_operating_days", "closing days", "clean_closing_days"]].head())


# ==========================
# STEP 4: PRICE PREP (OPTIONAL AVG PRICE)
# ==========================

print("\n🔄 Step 4: Preparing average price column...")

min_col = "min price (RM)"
max_col = "max price (RM)"

if min_col in df.columns and max_col in df.columns:
    # convert to numeric
    df[min_col] = pd.to_numeric(df[min_col], errors="coerce")
    df[max_col] = pd.to_numeric(df[max_col], errors="coerce")
    df["avg_price_rm"] = df[[min_col, max_col]].mean(axis=1)
else:
    df["avg_price_rm"] = None

print("✅ Step 4 done.")
print(df[[min_col, max_col, "avg_price_rm"]].head())


# ==========================
# STEP 5: BUILD COMBINED TEXT FIELD
# ==========================

print("\n🔄 Step 5: Building combined_text for TF-IDF/SBERT...")

def build_combined_text(row):
    title = str(row.get("title", "")).strip()
    tags = str(row.get("clean_tags", "")).strip()
    desc = str(row.get("clean_description", "")).strip()
    return " ".join([part for part in [title, tags, desc] if part])

df["combined_text"] = df.apply(build_combined_text, axis=1)

print("✅ Step 5 done.")
print(df[["title", "clean_tags", "clean_description", "combined_text"]].head())


# ==========================
# STEP 6: SAVE CLEANED METADATA
# ==========================

print("\n🔄 Step 6: Saving cleaned metadata...")

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False, encoding="utf-8")

print("🎉 ALL DONE!")
print("Cleaned metadata saved to:")
print(output_path)
