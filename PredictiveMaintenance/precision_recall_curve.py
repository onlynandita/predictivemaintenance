import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

from imblearn.over_sampling import SMOTE


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("dataset/ai4i2020.csv")

df.drop(
    columns=["UDI", "Product ID", "Air temperature [K]"],
    inplace=True
)

le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

features = [
    "Type",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


# =====================================================
# 2. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# 3. APPLY SMOTE TO TRAINING DATA ONLY
# =====================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# =====================================================
# 4. DEFINE MODELS
# =====================================================

models = {
    "Random Forest - Original": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),
        X_train,
        y_train
    ),

    "Random Forest - SMOTE": (
        RandomForestClassifier(
            random_state=42
        ),
        X_train_smote,
        y_train_smote
    ),

    "Decision Tree - Original": (
        DecisionTreeClassifier(
            random_state=42
        ),
        X_train,
        y_train
    ),

    "Decision Tree - SMOTE": (
        DecisionTreeClassifier(
            random_state=42
        ),
        X_train_smote,
        y_train_smote
    ),

    "Logistic Regression - Original": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        ),
        X_train,
        y_train
    ),

    "Logistic Regression - SMOTE": (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        ),
        X_train_smote,
        y_train_smote
    )
}


# =====================================================
# 5. PLOT PRECISION-RECALL CURVES
# =====================================================

os.makedirs("evaluation_results", exist_ok=True)

plt.figure(figsize=(11, 7))

print("\nPRECISION-RECALL EVALUATION")
print("=" * 60)

for name, (model, X_fit, y_fit) in models.items():

    print(f"Evaluating: {name}")

    model.fit(X_fit, y_fit)

    # Probability of machine failure (class 1)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        probabilities
    )

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP={average_precision:.3f})"
    )

    print(f"Average Precision: {average_precision:.4f}")


# Baseline equals the positive-class proportion in the test set
baseline = y_test.mean()

plt.axhline(
    baseline,
    linestyle="--",
    label=f"Baseline ({baseline:.3f})"
)

plt.title("Precision-Recall Curves - Predictive Maintenance")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend(loc="best", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = (
    "evaluation_results/precision_recall_curve.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nGraph saved to: {output_path}")

plt.show()
plt.close()