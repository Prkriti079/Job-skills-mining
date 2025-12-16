import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pathlib import Path
import os
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
INPUT_FILE_NAME = "tokens_by_job_id.csv"
MAX_K_FOR_ELBOW = 20  # Maximum K value to test for the Elbow Method
FINAL_K_CLUSTERS = 13  # The K value selected for final clustering

try:
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    script_dir = Path(os.getcwd())

INPUT_DIR = script_dir / "Input"
OUTPUT_DATA_DIR = script_dir / "output"
IMAGES_DIR = script_dir / "images"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = INPUT_DIR / INPUT_FILE_NAME

print(f"Attempting to load data from: {INPUT_FILE_PATH}...")

# Load the token data
try:
    df_raw = pd.read_csv(INPUT_FILE_PATH)
except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Input file not found.")
    print(
        f"ACTION REQUIRED: Please ensure your file '{INPUT_FILE_NAME}' is placed inside the '{INPUT_DIR.name}' folder."
    )
    exit()

# Filter for computer science tokens only
df_filtered = df_raw[df_raw["is_compsci"] == True].copy()
if df_filtered.empty:
    print("No computer science tokens found. Cannot perform analysis.")
    exit()

# Aggregate tokens back into 'documents' (job descriptions) for TF-IDF
job_skills = (
    df_filtered.groupby("job_id")["token"].apply(lambda x: " ".join(x)).reset_index()
)
job_skills.rename(columns={"token": "skill_text"}, inplace=True)

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(job_skills["skill_text"])
X = tfidf_matrix.toarray()
feature_names = vectorizer.get_feature_names_out()


# 1. K-Means ELBOW METHOD (Determining optimal K)

print("\nStarting Elbow Method Analysis")
inertia_data = []

for k in range(1, MAX_K_FOR_ELBOW + 1):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_data.append({"K": k, "Inertia": kmeans.inertia_})

elbow_df = pd.DataFrame(inertia_data)

# Save Elbow Data CSV
elbow_df.to_csv(OUTPUT_DATA_DIR / "kmeans_elbow_data.csv", index=False)
print(f"Elbow data saved to: {OUTPUT_DATA_DIR / 'kmeans_elbow_data.csv'}")

# Plot Elbow Curve
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(elbow_df["K"], elbow_df["Inertia"], marker="o", linestyle="-", color="blue")
plt.title("Elbow Method for Optimal K", fontsize=16)
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(1, MAX_K_FOR_ELBOW + 1))

# Highlight the chosen K (FINAL_K_CLUSTERS)
plt.axvline(
    x=FINAL_K_CLUSTERS,
    color="red",
    linestyle="--",
    label=f"Chosen K={FINAL_K_CLUSTERS}",
)
plt.legend()

# Save Elbow Plot PNG
fig.savefig(IMAGES_DIR / "kmeans_elbow_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Elbow plot saved to: {IMAGES_DIR / 'kmeans_elbow_plot.png'}")

# 2. FINAL K-MEANS CLUSTERING (K=13)

print(f"\nRunning Final K-Means Clustering with K={FINAL_K_CLUSTERS}")
kmeans_final = KMeans(
    n_clusters=FINAL_K_CLUSTERS, init="k-means++", random_state=42, n_init=10
)
job_skills["cluster"] = kmeans_final.fit_predict(X)

# Save Job ID and Cluster assignments CSV
job_id_cluster_df = job_skills[["job_id", "cluster"]].copy()
job_id_cluster_df.to_csv(OUTPUT_DATA_DIR / "job_id_with_cluster.csv", index=False)
print(
    f"Job ID cluster assignments saved to: {OUTPUT_DATA_DIR / 'job_id_with_cluster.csv'}"
)

# 3. CLUSTER REPORT (Top Skills per Cluster)

print("\nGenerating Cluster Report (Top Skills)")
cluster_centers = kmeans_final.cluster_centers_
cluster_report_data = []

for i in range(FINAL_K_CLUSTERS):

    # Sort features by weight in the cluster center
    feature_weights = dict(zip(feature_names, cluster_centers[i]))
    top_skills_weights = sorted(
        feature_weights.items(), key=lambda item: item[1], reverse=True
    )

    cluster_size = len(job_skills[job_skills["cluster"] == i])

    top_skills_list = [
        f"{skill} ({weight:.3f})" for skill, weight in top_skills_weights[:15]
    ]

    cluster_report_data.append(
        {
            "Cluster ID": i,
            "Size": cluster_size,
            "Top 15 Skills (Token & Weight)": "; ".join(top_skills_list),
        }
    )

cluster_report_df = pd.DataFrame(cluster_report_data)

# Save K-Means Clustering Report CSV
cluster_report_df.to_csv(OUTPUT_DATA_DIR / "kmeans_clustering_report.csv", index=False)
print(
    f"K-Means cluster report saved to: {OUTPUT_DATA_DIR / 'kmeans_clustering_report.csv'}"
)

print("\n--- K-Means Analysis Complete ---")
