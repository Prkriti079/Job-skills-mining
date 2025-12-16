import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
CLUSTERS_FILE = "kmeans_job_clusters.csv"
TOP_SKILLS_FILE = "cluster_top_skills.csv"
RULES_FILE = "association_rules.csv"
IMAGES_DIR_NAME = "images"


def setup_paths():
    """Sets up project directory paths."""
    # Use the robust path setup using os.path.abspath(__file__)
    try:
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        PROJECT_ROOT = script_dir.parent
        OUTPUT_DIR = PROJECT_ROOT / "output"
        IMAGES_DIR = PROJECT_ROOT / IMAGES_DIR_NAME
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR, IMAGES_DIR
    except Exception as e:
        print(f"Error setting up paths: {e}")
        return None, None


def load_results(OUTPUT_DIR):
    """Loads the cluster and rule analysis results."""
    print("\n1. Loading results from Clustering and Association Mining...")
    try:
        clusters_df = pd.read_csv(OUTPUT_DIR / CLUSTERS_FILE)
        skills_df = pd.read_csv(OUTPUT_DIR / TOP_SKILLS_FILE)

        # Association rules might be empty if thresholds were too high
        try:
            rules_df = pd.read_csv(OUTPUT_DIR / RULES_FILE)
        except FileNotFoundError:
            print(
                "Warning: Association rules file not found. Skipping rule visualization."
            )
            rules_df = None

        print("   -> All required result files loaded.")
        return clusters_df, skills_df, rules_df
    except FileNotFoundError as e:
        print(
            f"CRITICAL ERROR: Could not find required result file: {e}. Aborting visualization."
        )
        print("Please ensure you have run steps 1, 2, and 3 successfully.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return None, None, None


def visualize_clusters(clusters_df, skills_df, IMAGES_DIR):
    """Generates visualizations for the K-Means Clustering results."""
    print("\n2. Generating Cluster Analysis Visualization...")

    # 1. Cluster Size Distribution
    cluster_counts = clusters_df["cluster"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
    plt.title("Distribution of Computer Science Jobs Across Clusters", fontsize=16)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Number of Jobs", fontsize=12)
    plt.grid(axis="y", linestyle="--")

    # Add text labels for counts on top of bars
    for i, count in enumerate(cluster_counts.values):
        plt.text(
            i,
            count + (cluster_counts.max() * 0.01),
            f"{count}",
            ha="center",
            va="bottom",
        )

    cluster_size_path = IMAGES_DIR / "cluster_job_distribution.png"
    plt.savefig(cluster_size_path)
    plt.close()
    print(f"✅ Saved Cluster Distribution plot to: {cluster_size_path.resolve()}")

    # 2. Print Top Skills Report (Replaces complex plotting attempt)
    print("\n   -> Top Skills per Cluster (Report from cluster_top_skills.csv):")
    # FIX: Column is 'Cluster ID', not 'Cluster'
    for index, row in skills_df.sort_values(by="Size", ascending=False).iterrows():
        top_skills = row["Top 15 Skills (Token & Weight)"].split("; ")
        # Print the top 3 for a concise report
        print(f"      Cluster {row['Cluster ID']} (Size: {row['Size']})")
        print(f"         - 1st: {top_skills[0].strip()}")
        print(f"         - 2nd: {top_skills[1].strip()}")
        print(f"         - 3rd: {top_skills[2].strip()}")


def visualize_association_rules(rules_df, IMAGES_DIR):
    """Generates visualizations for the Association Rules results."""
    print("\n3. Generating Association Rule Visualizations...")

    if rules_df is None or rules_df.empty:
        print("   -> Skipped: No association rules found or loaded.")
        return

    # Pre-process the rules for plotting
    rules_plot = rules_df.copy()

    # Robustly handle string conversion for frozensets in CSV
    def get_antecedent_len(s):
        # Remove brackets and quotes, then split by comma and count items
        s = str(s).replace("{", "").replace("}", "").replace("'", "").strip()
        if not s:
            return 0
        return len(s.split(","))

    rules_plot["antecedent_len"] = rules_plot["antecedents"].apply(get_antecedent_len)

    # Filter for the top 50 rules by Lift for a manageable plot
    rules_plot = rules_plot.sort_values(by="lift", ascending=False).head(50)

    # 3. Scatter Plot of Rules (Support vs. Confidence, sized by Lift)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="support",
        y="confidence",
        size="lift",
        hue="antecedent_len",
        data=rules_plot,
        sizes=(50, 500),
        palette="viridis",
        alpha=0.7,
    )

    plt.title("Top 50 Association Rules: Support, Confidence, and Lift", fontsize=16)
    plt.xlabel("Support (Fraction of Total Jobs)", fontsize=12)
    plt.ylabel("Confidence (P(Consequent | Antecedent))", fontsize=12)
    plt.legend(title="Antecedent Size", loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)

    rules_scatter_path = IMAGES_DIR / "rules_scatter_plot.png"
    plt.savefig(rules_scatter_path)
    plt.close()
    print(f"✅ Saved Association Rules Scatter plot to: {rules_scatter_path.resolve()}")


def run_pipeline():
    OUTPUT_DIR, IMAGES_DIR = setup_paths()
    if not OUTPUT_DIR:
        return

    clusters_df, skills_df, rules_df = load_results(OUTPUT_DIR)

    if clusters_df is not None and skills_df is not None:
        visualize_clusters(clusters_df, skills_df, IMAGES_DIR)

    if rules_df is not None:
        visualize_association_rules(rules_df, IMAGES_DIR)

    print("\n--- Data Mining Pipeline COMPLETE ---")
    print(
        f"You have successfully generated all data products. Check the 'output' folder for CSVs and the '{IMAGES_DIR_NAME}' folder for all required visualizations for your final report/dashboard."
    )


if __name__ == "__main__":
    run_pipeline()
