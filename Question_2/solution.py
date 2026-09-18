import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# QUESTION 2 - TWELVE THOUSAND TENDERS
# PART (A) - DEFINE SIMILARITY MECHANICALLY
# ============================================================

DATA_PATH = "."

# ------------------------------------------------------------
# 1. LOAD ALL NOTICE FILES
# ------------------------------------------------------------

notice_files = glob.glob(
    os.path.join(DATA_PATH, "notices", "*.csv")
)

notice_list = []

for file in notice_files:
    temp = pd.read_csv(file)
    notice_list.append(temp)

notices = pd.concat(
    notice_list,
    ignore_index=True
)

print("=" * 70)
print("QUESTION 2 - TENDER DUPLICATE DETECTION")
print("=" * 70)

print("\nTotal notices:", len(notices))

print("\nColumns:")
print(notices.columns.tolist())


# ------------------------------------------------------------
# 2. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Replace non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


notices["title_clean"] = (
    notices["title"]
    .fillna("")
    .apply(clean_text)
)

notices["body_clean"] = (
    notices["body"]
    .fillna("")
    .apply(clean_text)
)

# Combine title and body
notices["text"] = (
    notices["title_clean"] +
    " " +
    notices["body_clean"]
)

print("\nText preprocessing completed.")

print("\nExample of cleaned text:")
print(notices[["title", "text"]].head(3))


# ------------------------------------------------------------
# 3. TF-IDF REPRESENTATION
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    notices["text"]
)

print("\nTF-IDF representation created.")
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ------------------------------------------------------------
# 4. NOTICE ID INDEX
# ------------------------------------------------------------

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


# ------------------------------------------------------------
# 5. LOAD LABELLED PAIRS
# ------------------------------------------------------------

pairs = pd.read_csv(
    os.path.join(DATA_PATH, "labelled_pairs.csv")
)

print("\nNumber of labelled pairs:", len(pairs))

print("\nLabel distribution:")
print(pairs["label"].value_counts())


# ------------------------------------------------------------
# 6. CALCULATE COSINE SIMILARITY
# ------------------------------------------------------------

scores = []

for _, row in pairs.iterrows():

    id_a = str(row["notice_id_a"])
    id_b = str(row["notice_id_b"])

    if id_a in notice_index and id_b in notice_index:

        index_a = notice_index[id_a]
        index_b = notice_index[id_b]

        score = cosine_similarity(
            tfidf_matrix[index_a],
            tfidf_matrix[index_b]
        )[0][0]

        scores.append(score)

    else:
        scores.append(np.nan)


pairs["similarity"] = scores

pairs_eval = pairs.dropna(
    subset=["similarity"]
).copy()

print("\nPairs successfully evaluated:",
      len(pairs_eval))


# ------------------------------------------------------------
# 7. CHECK SIMILARITY DISTRIBUTION
# ------------------------------------------------------------

print("\nAverage similarity by label:")

print(
    pairs_eval
    .groupby("label")["similarity"]
    .mean()
)


# ------------------------------------------------------------
# 8. PLOT SAME VS DIFFERENT SIMILARITY
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

same_scores = pairs_eval[
    pairs_eval["label"] == "same"
]["similarity"]

different_scores = pairs_eval[
    pairs_eval["label"] == "different"
]["similarity"]

plt.hist(
    same_scores,
    bins=30,
    alpha=0.6,
    label="Same"
)

plt.hist(
    different_scores,
    bins=30,
    alpha=0.6,
    label="Different"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title("Similarity Score Distribution: Same vs Different")
plt.legend()
plt.grid(True)

plt.savefig(
    "similarity_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 9. TEST DIFFERENT THRESHOLDS
# ------------------------------------------------------------

pairs_eval["actual"] = (
    pairs_eval["label"]
    .str.lower()
    .eq("same")
    .astype(int)
)

threshold_results = []

for threshold in np.arange(0.10, 0.96, 0.05):

    predicted = (
        pairs_eval["similarity"] >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        pairs_eval["actual"],
        predicted
    )

    precision = precision_score(
        pairs_eval["actual"],
        predicted,
        zero_division=0
    )

    recall = recall_score(
        pairs_eval["actual"],
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        pairs_eval["actual"],
        predicted,
        zero_division=0
    )

    threshold_results.append([
        threshold,
        accuracy,
        precision,
        recall,
        f1
    ])


results_df = pd.DataFrame(
    threshold_results,
    columns=[
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)

print("\n")
print("=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 10. FIND BEST F1 THRESHOLD
# ------------------------------------------------------------

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = best_row["threshold"]

print("\nBest threshold based on F1:")
print(best_threshold)

print("Accuracy:", best_row["accuracy"])
print("Precision:", best_row["precision"])
print("Recall:", best_row["recall"])
print("F1:", best_row["f1"])


# ------------------------------------------------------------
# 11. PLOT THRESHOLD PERFORMANCE
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["threshold"],
    results_df["accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["threshold"],
    results_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1"],
    marker="o",
    label="F1 Score"
)

plt.xlabel("Similarity Threshold")
plt.ylabel("Score")
plt.title("Threshold vs Classification Performance")
plt.legend()
plt.grid(True)

plt.savefig(
    "threshold_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 12. FINAL CLASSIFICATION
# ------------------------------------------------------------

pairs_eval["predicted"] = (
    pairs_eval["similarity"] >= best_threshold
).astype(int)

print("\n")
print("=" * 70)
print("FINAL PERFORMANCE")
print("=" * 70)

print(
    "Accuracy :",
    accuracy_score(
        pairs_eval["actual"],
        pairs_eval["predicted"]
    )
)

print(
    "Precision:",
    precision_score(
        pairs_eval["actual"],
        pairs_eval["predicted"],
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        pairs_eval["actual"],
        pairs_eval["predicted"],
        zero_division=0
    )
)

print(
    "F1 Score :",
    f1_score(
        pairs_eval["actual"],
        pairs_eval["predicted"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 13. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    pairs_eval["actual"],
    pairs_eval["predicted"]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Different", "Same"]
)

disp.plot()

plt.title("Confusion Matrix")
plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n")
print("=" * 70)
print("PART (A) COMPLETED")
print("=" * 70)