import pandas as pd

# IMPORTANT CHANGE: Switched from apriori to fpgrowth for memory efficiency
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pathlib import Path
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# --- Configuration ---
INPUT_FILE_NAME = "tokens_by_job_id.csv"
OUTPUT_RULES_FILE = "association_rules.csv"

# Association Rule Hyperparameters
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.2

# Token Pre-filtering Threshold (Kept at 10000 to maintain small feature set)
TOKEN_FREQUENCY_THRESHOLD = 10000


def load_data_and_prepare_market_basket(input_path):
    """
    Loads the tokenized data and transforms it into the Market Basket format
    (one-hot encoded DataFrame, where rows are job_ids and columns are tokens).
    """
    print(f"1. Loading transaction data from: {input_path.resolve()}")
    try:
        # Load the preprocessed token data
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(
            f"ERROR: Input file not found at {input_path}. Please run data_preprocessing_pipeline.py first."
        )
        return None, None

    # Filter only for Computer Science jobs (is_compsci == True)
    df_compsci = df[df["is_compsci"] == True].copy()
    total_jobs = len(df_compsci["job_id"].unique())
    print(f"   -> Successfully loaded {total_jobs} unique Computer Science jobs.")

    # --- CRITICAL STEP: Filter out low-frequency tokens to prevent memory overflow ---
    print(
        f"   -> Filtering tokens that appear in fewer than {TOKEN_FREQUENCY_THRESHOLD} jobs..."
    )

    # 1. Count how many unique jobs each token appears in
    token_job_counts = df_compsci.groupby("token")["job_id"].nunique()

    # 2. Get the list of tokens that meet the threshold
    frequent_tokens = token_job_counts[
        token_job_counts >= TOKEN_FREQUENCY_THRESHOLD
    ].index

    # 3. Filter the main DataFrame to keep only frequent tokens
    df_compsci_filtered = df_compsci[df_compsci["token"].isin(frequent_tokens)].copy()

    tokens_kept = len(frequent_tokens)
    print(f"   -> Kept {tokens_kept} tokens (columns) for FP-Growth analysis.")

    if tokens_kept == 0:
        print("ERROR: Zero tokens remaining. Increase TOKEN_FREQUENCY_THRESHOLD.")
        return None, None

    # Create the transaction dataset (Market Basket format)
    # The 'size' column is a count of a token in a job, but we only care about presence (1)
    basket_sets = (
        df_compsci_filtered.groupby(["job_id", "token"])["size"]
        .sum()
        .unstack(fill_value=0)
    )

    # Convert token counts to a binary (0/1) presence map
    def encode_units(x):
        return 1 if x >= 1 else 0

    # Ensure the dataframe is in boolean format for the most efficient processing by mlxtend
    basket_encoded = basket_sets.applymap(encode_units).astype(bool)

    return basket_encoded, df_compsci


def run_fpgrowth_analysis(basket_encoded):
    """
    Runs the FP-Growth algorithm and generates association rules.

    """
    print("\n2. Running FP-Growth Algorithm...")
    print(f"   -> Min Support: {MIN_SUPPORT}")
    print(f"   -> Min Confidence: {MIN_CONFIDENCE}")
    print(f"   -> Min Lift: {MIN_LIFT}")
    print(f"   -> Note: FP-Growth is used to bypass memory limits of Apriori.")

    # Step 1: Find Frequent Itemsets using FP-Growth
    # FP-Growth is highly optimized for memory and speed, especially on sparse data.
    frequent_itemsets = fpgrowth(
        basket_encoded,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        # max_len is omitted, but we will filter by length 2 after generation
    )

    # Filter itemsets to only include pairs (length 2), matching the original goal
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets[frequent_itemsets["length"] == 2]

    print(f"   -> Found {len(frequent_itemsets)} frequent itemsets of length 2.")

    # Step 2: Generate Association Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=MIN_LIFT)

    # Filter rules by minimum confidence after the lift filter
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE]

    # Clean up and sort the results
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

    return rules


def run_pipeline():
    """Main execution function for the association mining pipeline."""

    try:
        # Determine paths
        script_path = Path(__file__).resolve()
        PROJECT_ROOT = script_path.parent.parent
        INPUT_PATH = PROJECT_ROOT / "output" / INPUT_FILE_NAME
        OUTPUT_PATH = PROJECT_ROOT / "output" / OUTPUT_RULES_FILE

    except Exception as e:
        print(f"Path initialization error: {e}")
        return

    # 1. Load data and prepare for FP-Growth
    basket_encoded, df_compsci = load_data_and_prepare_market_basket(INPUT_PATH)

    if basket_encoded is None:
        return

    # 2. Run FP-Growth analysis
    rules = run_fpgrowth_analysis(basket_encoded)

    if rules.empty:
        print("\n--- FP-Growth Complete ---")
        print("❌ No association rules found based on the current thresholds.")
        print(
            "Try lowering MIN_SUPPORT (e.g., to 0.005) or MIN_CONFIDENCE (e.g., to 0.4) and re-running the script."
        )
        return

    # 3. Save Results
    rules.to_csv(OUTPUT_PATH, index=False)

    print("\n3. Results Summary:")
    print(f"   -> Total strong rules found: {len(rules)}")
    print(f"   -> Rules saved to: {OUTPUT_PATH.resolve()}")

    print("\n--- Top 10 Association Rules (Strongest Lift) ---")
    # Using max_rows=None ensures all columns are printed if they fit the console width
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]]
            .head(10)
            .to_markdown(index=False)
        )

    print("\n--- FP-Growth Analysis Complete ---")


if __name__ == "__main__":
    run_pipeline()
