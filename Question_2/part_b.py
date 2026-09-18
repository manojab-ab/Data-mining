import pandas as pd
import numpy as np
import glob
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# QUESTION 2(b) - REDUCED REPRESENTATION
# ============================================================

DATA_PATH = "."


# ------------------------------------------------------------
# 1. LOAD NOTICES
# ------------------------------------------------------------

notice_files = glob.glob(
    os.path.join(DATA_PATH, "notices", "*.csv")
)

notice_list = []

for file in notice_files:
    notice_list.append(pd.read_csv(file))

notices = pd.concat(
    notice_list,
    ignore_index=True
)

print("=" * 70)
print("QUESTION 2(b) - REDUCED REPRESENTATION")
print("=" * 70)

print("\nTotal notices:", len(notices))


# ------------------------------------------------------------
# 2. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


notices["text"] = (
    notices["title"].fillna("").apply(clean_text)
    + " "
    + notices["body"].fillna("").apply(clean_text)
)


# ------------------------------------------------------------
# 3. ORIGINAL TF-IDF
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    notices["text"]
)

print("\nOriginal representation:")
print("TF-IDF shape:", tfidf_matrix.shape)


# ------------------------------------------------------------
# 4. REDUCE TO 100 DIMENSIONS
# ------------------------------------------------------------

svd = TruncatedSVD(
    n_components=100,
    random_state=42
)

reduced_matrix = svd.fit_transform(
    tfidf_matrix
)

print("\nReduced representation:")
print("SVD shape:", reduced_matrix.shape)

print(
    "Explained variance:",
    svd.explained_variance_ratio_.sum()
)


# ------------------------------------------------------------
# 5. NOTICE ID INDEX
# ------------------------------------------------------------

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


# ------------------------------------------------------------
# 6. LOAD LABELLED PAIRS
# ------------------------------------------------------------

pairs = pd.read_csv(
    os.path.join(DATA_PATH, "labelled_pairs.csv")
)

pairs["actual"] = (
    pairs["label"]
    .str.lower()
    .eq("same")
    .astype(int)
)


# ------------------------------------------------------------
# 7. CALCULATE REDUCED-SPACE SIMILARITY
# ------------------------------------------------------------

scores = []

for _, row in pairs.iterrows():

    id_a = str(row["notice_id_a"])
    id_b = str(row["notice_id_b"])

    index_a = notice_index[id_a]
    index_b = notice_index[id_b]

    score = cosine_similarity(
        reduced_matrix[index_a].reshape(1, -1),
        reduced_matrix[index_b].reshape(1, -1)
    )[0][0]

    scores.append(score)


pairs["similarity"] = scores


# ------------------------------------------------------------
# 8. TEST THRESHOLDS
# ------------------------------------------------------------

threshold_results = []

for threshold in np.arange(0.10, 0.96, 0.05):

    predicted = (
        pairs["similarity"] >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        pairs["actual"],
        predicted
    )

    precision = precision_score(
        pairs["actual"],
        predicted,
        zero_division=0
    )

    recall = recall_score(
        pairs["actual"],
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        pairs["actual"],
        predicted,
        zero_division=0
    )

    threshold_results.append(
        [
            threshold,
            accuracy,
            precision,
            recall,
            f1
        ]
    )


results = pd.DataFrame(
    threshold_results,
    columns=[
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)


# ------------------------------------------------------------
# 9. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("REDUCED REPRESENTATION THRESHOLD RESULTS")
print("=" * 70)

print(
    results.to_string(index=False)
)


# ------------------------------------------------------------
# 10. BEST RESULT
# ------------------------------------------------------------

best = results.loc[
    results["f1"].idxmax()
]

print("\n")
print("=" * 70)
print("BEST REDUCED REPRESENTATION RESULT")
print("=" * 70)

print("Threshold :", best["threshold"])
print("Accuracy  :", best["accuracy"])
print("Precision :", best["precision"])
print("Recall    :", best["recall"])
print("F1 Score  :", best["f1"])


# ------------------------------------------------------------
# 11. COMPRESSION INFORMATION
# ------------------------------------------------------------

original_dimensions = 5000
reduced_dimensions = 100

reduction_percentage = (
    1 - reduced_dimensions / original_dimensions
) * 100

print("\n")
print("=" * 70)
print("SPACE REDUCTION")
print("=" * 70)

print(
    "Original dimensions:",
    original_dimensions
)

print(
    "Reduced dimensions:",
    reduced_dimensions
)

print(
    "Dimensionality reduction:",
    f"{reduction_percentage:.2f}%"
)

print("\n")
print("=" * 70)
print("PART (B) COMPLETED")
print("=" * 70)