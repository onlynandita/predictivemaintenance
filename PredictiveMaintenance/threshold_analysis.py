import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/ai4i2020.csv")

df.drop(
    columns=["UDI", "Product ID", "Air temperature [K]"],
    inplace=True
)

encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])

features = [
    "Type",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. TRAIN ORIGINAL RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# 4. TEST DIFFERENT THRESHOLDS
# ==========================================

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

results = []

print("\nRANDOM FOREST THRESHOLD ANALYSIS")
print("=" * 85)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-score (%)": round(f1 * 100, 2),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    })

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# ==========================================
# 5. SAVE RESULTS
# ==========================================

os.makedirs("evaluation_results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation_results/threshold_analysis.csv",
    index=False
)

print("\nComplete Threshold Comparison:")
print(results_df.to_string(index=False))

print(
    "\nResults saved to "
    "evaluation_results/threshold_analysis.csv"
)


# ==========================================
# 6. PLOT RECALL AND PRECISION
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Recall (%)"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision (%)"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1-score (%)"],
    marker="o",
    label="F1-score"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score (%)")

plt.title(
    "Random Forest Threshold Optimization"
)

plt.xticks(thresholds)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "evaluation_results/threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Graph saved to "
    "evaluation_results/threshold_analysis.png"
)

plt.show()
plt.close()