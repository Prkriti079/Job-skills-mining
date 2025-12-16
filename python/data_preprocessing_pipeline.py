import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
import warnings
import nltk
import os

# --- Initial Setup ---
try:
    # Ensure NLTK resources are available
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception:
    print(
        "Warning: Failed to download NLTK resources. Please ensure NLTK is installed."
    )

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration and File Names
POSTINGS_FILE = "postings.csv"
INDUSTRIES_FILE = "job_industries.csv"
SKILLS_FILE = "job_skills.csv"
OUTPUT_FILE_NAME = "tokens_by_job_id.csv"

# NLP Filters and Keywords
STOP_WORDS = set(stopwords.words("english"))

# IDs for technology-related industries
TECH_INDUSTRY_IDS = [
    11,
    12,
    13,
    14,
    16,
    94,
    96,
    100,
    102,
    103,
    107,
    108,
    109,
    110,
    111,
    113,
    114,
]
KEYWORD_FILTER = [
    "data science",
    "machine learning",
    "ai",
    "artificial intelligence",
    "deep learning",
    "computer science",
    "software engineer",
    "developer",
    "programmer",
    "engineer",
    "cyber security",
    "network",
    "cloud computing",
    "web development",
    "full stack",
]
keyword_pattern = "|".join([re.escape(k) for k in KEYWORD_FILTER])


def clean_and_tokenize(text):
    """Cleans text, tokenizes, and removes stop words and non-alphanumeric tokens."""
    if pd.isna(text) or text is None:
        return []

    # Simple cleanup: remove punctuation and numbers, convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", str(text)).lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Filter out short tokens, stop words, and non-alphanumeric words
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2 and token.isalnum()
    ]

    return tokens


def run_pipeline():
    """Executes the data cleaning, merging, and tokenization pipeline."""

    # irectory Setup for Project Structure
    try:
        # 1. Get the directory of the current script
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(os.getcwd())

    PROJECT_ROOT = script_dir.parent

    # 3. Define data folder paths based on the project root
    INPUT_DATA_DIR = PROJECT_ROOT / "input"
    OUTPUT_DATA_DIR = PROJECT_ROOT / "output"

    # 1. Load Data
    print("1. Loading raw data files...")

    postings_path = INPUT_DATA_DIR / POSTINGS_FILE

    # Critical Path Check
    print(f"Verifying expected path for '{POSTINGS_FILE}': {postings_path.resolve()}")

    if not postings_path.exists():
        print("\n=====================================================================")
        print("CRITICAL FILE NOT FOUND ERROR (CHECK FILE SYSTEM):")
        print(f"The script cannot find the file at the path printed above.")
        print("Please verify the file path and try again.")
        print("=====================================================================")
        return

    try:
        # Load the files using the constructed path
        df_postings = pd.read_csv(postings_path)
        df_industries = pd.read_csv(INPUT_DATA_DIR / INDUSTRIES_FILE)
        df_skills = pd.read_csv(INPUT_DATA_DIR / SKILLS_FILE)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Input file not found: {e}")
        print(f"Please ensure all three files are in the 'input' folder.")
        return

    # 2. MERGE: 'industry_id'
    print("2. Merging postings with industry data...")

    # Use 'how=left' to keep all job postings even if an industry ID is missing
    df_postings = pd.merge(
        df_postings, df_industries[["job_id", "industry_id"]], on="job_id", how="left"
    )

    # Delete unused raw dataframes to free up memory
    del df_industries
    del df_skills

    # 3. Prepare Data Frames
    print("3. Filtering for relevant jobs...")

    # Prepare the 'industry_id' column, filling missing values (NaN after merge) with 0
    df_postings["industry_id"] = (
        pd.to_numeric(df_postings["industry_id"], errors="coerce").fillna(0).astype(int)
    )

    # Merge job text (title and description) into one column
    df_postings["full_text"] = (
        df_postings["title"].astype(str) + " " + df_postings["description"].astype(str)
    )

    # 4. Filter for Computer Science / Tech Related Jobs
    is_industry_tech = df_postings["industry_id"].isin(TECH_INDUSTRY_IDS)
    is_keyword_tech = df_postings["full_text"].str.contains(
        keyword_pattern, case=False, na=False
    )
    df_postings["is_compsci"] = is_industry_tech | is_keyword_tech
    df_filtered = df_postings[df_postings["is_compsci"] == True].copy()

    # Free up the original postings dataframe
    del df_postings

    # 5. Clean, Tokenize, and Explode
    print(
        "4. Cleaning, tokenizing, and exploding data into final transaction format (This step is slow, please wait)..."
    )

    df_filtered["tokens"] = df_filtered["full_text"].apply(clean_and_tokenize)

    # Free up memory used by the large text column
    del df_filtered["full_text"]

    token_df = df_filtered.explode("tokens").rename(columns={"tokens": "token"})
    token_df = token_df.dropna(subset=["token"]).copy()

    # Free up the filtered dataframe
    del df_filtered

    # Aggregate to get token size (frequency) per job
    final_df = (
        token_df.groupby(["job_id", "token", "industry_id", "is_compsci"])
        .size()
        .reset_index(name="size")
    )

    # Free up the exploded token dataframe
    del token_df

    # 6. Save the final file
    OUTPUT_FILE_PATH = OUTPUT_DATA_DIR / OUTPUT_FILE_NAME
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)

    print(f"\n--- Preprocessing Complete ---")
    print(f"Final input file generated: {OUTPUT_FILE_PATH.resolve()}")
    print(f"Total tokens/transactions generated: {len(final_df)}.")


if __name__ == "__main__":
    run_pipeline()
