import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


plt.style.use("seaborn-v0_8-whitegrid")


def clean_text(text):
    text = str(text).lower()
    text = text.replace("<", " ").replace(">", " ")
    text = text.replace("\n", " ")
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = " ".join(text.split())
    return text


# 1) Label distribution chart
pairs = pd.read_csv("labelled_pairs.csv")
label_counts = pairs["label"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#1f77b4", "#d62728"]
ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor="black", linewidth=1.2)
ax.set_title("Label Distribution", fontsize=14, weight="bold")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
for bar, value in zip(ax.patches, label_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig("label_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# 2) Top portal distribution chart
notice_files = glob.glob(os.path.join("notices", "*.csv"))
frames = [pd.read_csv(f) for f in notice_files]
notices = pd.concat(frames, ignore_index=True)
portal_counts = notices["portal_id"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
palette = plt.cm.Blues(np.linspace(0.35, 0.95, len(portal_counts)))
ax.barh(portal_counts.index[::-1], portal_counts.values[::-1], color=palette, edgecolor="black")
ax.set_title("Top 10 Portals by Notice Count", fontsize=14, weight="bold")
ax.set_xlabel("Number of Notices")
ax.set_ylabel("Portal ID")
fig.tight_layout()
fig.savefig("portal_distribution_top10.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# 3) Threshold performance chart from recorded output values
thresholds = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
accuracy = np.array([0.353333, 0.520000, 0.696667, 0.786667, 0.870000, 0.946667, 0.978889, 0.986667, 0.988889, 0.992222, 0.994444, 0.987778, 0.981111, 0.957778, 0.924444, 0.888889, 0.832222, 0.788889])
precision = np.array([0.324042, 0.392405, 0.505435, 0.592357, 0.704545, 0.853211, 0.936242, 0.958763, 0.965398, 0.975524, 0.989286, 0.992647, 0.996212, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000])
recall = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.992832, 0.967742, 0.942652, 0.863799, 0.756272, 0.641577, 0.458781, 0.318996])
f1 = np.array([0.489474, 0.563636, 0.671480, 0.744000, 0.826667, 0.920792, 0.967071, 0.978947, 0.982394, 0.987611, 0.991055, 0.980036, 0.968692, 0.926923, 0.861224, 0.781659, 0.628993, 0.483696])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, accuracy, marker="o", linewidth=2.5, color="#2E86DE", label="Accuracy")
ax.plot(thresholds, precision, marker="s", linewidth=2.5, color="#28B463", label="Precision")
ax.plot(thresholds, recall, marker="D", linewidth=2.5, color="#F39C12", label="Recall")
ax.plot(thresholds, f1, marker="^", linewidth=2.5, color="#E74C3C", label="F1 Score")
ax.set_title("Threshold Sensitivity Analysis", fontsize=14, weight="bold")
ax.set_xlabel("Similarity Threshold")
ax.set_ylabel("Score")
ax.legend(loc="best")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("threshold_performance_colored.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Created: label_distribution.png")
print("Created: portal_distribution_top10.png")
print("Created: threshold_performance_colored.png")
