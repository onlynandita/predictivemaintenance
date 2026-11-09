import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("dataset/ai4i2020.csv")

df.drop(
    columns=[
        "UDI",
        "Product ID",
        "Air temperature [K]"
    ],
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
# 3. CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs("evaluation_results", exist_ok=True)


# ==========================================
# 4. ORIGINAL RANDOM FOREST
# ==========================================

original_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

original_model.fit(X_train, y_train)

original_predictions = original_model.predict(X_test)

original_cm = confusion_matrix(
    y_test,
    original_predictions,
    labels=[0, 1]
)


# ==========================================
# 5. SMOTE RANDOM FOREST
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

smote_model = RandomForestClassifier(
    random_state=42
)

smote_model.fit(X_train_smote, y_train_smote)

smote_predictions = smote_model.predict(X_test)

smote_cm = confusion_matrix(
    y_test,
    smote_predictions,
    labels=[0, 1]
)


# ==========================================
# 6. PLOT BOTH CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

matrices = [
    (
        original_cm,
        "Random Forest - Original Data",
        axes[0]
    ),
    (
        smote_cm,
        "Random Forest - SMOTE",
        axes[1]
    )
]

for matrix, title, ax in matrices:

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Failure"],
        yticklabels=["Healthy", "Failure"],
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.suptitle(
    "Confusion Matrix Comparison - Predictive Maintenance"
)

plt.tight_layout()

output_path = (
    "evaluation_results/"
    "random_forest_confusion_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("Original Random Forest:")
print(original_cm)

print("\nSMOTE Random Forest:")
print(smote_cm)

print(f"\nConfusion matrix graph saved to: {output_path}")

plt.show()
plt.close()