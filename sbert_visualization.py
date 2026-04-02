# sbert_visualization.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# ============================
# PATHS
# ============================
metadata_path = r"C:\laragon\www\web\lokalkita\model\clean_data\metadata\metadata_cleaned.csv"
embeddings_path = r"C:\laragon\www\web\lokalkita\model\metadata_model\metadata_sbert_embeddings.npy"

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(metadata_path)
emb = np.load(embeddings_path)

print("Metadata:", df.shape)
print("Embeddings:", emb.shape)

# Required column
if "category" not in df.columns:
    raise ValueError("❌ Column 'category' not found in metadata.")

# ============================
# PCA REDUCTION (384 → 2D)
# ============================
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb)

df["pc1"] = emb_2d[:, 0]
df["pc2"] = emb_2d[:, 1]

# ============================
# COLOR CODING BY CATEGORY
# ============================
categories = df["category"].unique()
num_categories = len(categories)

colors = cm.get_cmap("tab20", num_categories)
category_color_map = {cat: colors(i) for i, cat in enumerate(categories)}

# ============================
# PLOT
# ============================
plt.figure(figsize=(12, 9))

for cat in categories:
    subset = df[df["category"] == cat]
    plt.scatter(
        subset["pc1"],
        subset["pc2"],
        s=40,
        color=category_color_map[cat],
        label=cat,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5
    )

# ============================
# AESTHETICS
# ============================
plt.title("SBERT Embedding Clusters (PCA 2D Visualization)", fontsize=16, weight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(alpha=0.25)

# Legend
plt.legend(
    title="Categories",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=10
)

plt.tight_layout()
plt.show()
