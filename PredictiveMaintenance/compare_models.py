import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

DATASET_PATH = os.path.join("dataset", "ai4i2020.csv")
OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# =========================================================
# 2. LOAD DATASET
# =========================================================

print("=" * 65)
print("IOCL PREDICTIVE MAINTENANCE - MODEL COMPARISON")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# =========================================================
# 3. DATA PREPROCESSING
# =========================================================

# Remove identifier columns
df.drop(
    columns=["UDI", "Product ID"],
    inplace=True,
    errors="ignore"
)

# Remove the same feature excluded in the original training script
df.drop(
    columns=["Air temperature [K]"],
    inplace=True,
    errors="ignore"
)

# Encode machine Type
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

print("\nMachine Type encoding:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Select input features
feature_columns = [
    "Type",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target_column = "Machine failure"

X = df[feature_columns].copy()
y = df[target_column].copy()

print("\nOriginal target distribution:")
print(y.value_counts().sort_index())

print("\nClass percentages:")
print(
    (y.value_counts(normalize=True).sort_index() * 100)
    .round(2)
    .astype(str)
    + "%"
)


# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================

# Stratification preserves the original class proportions
# in both training and test sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining distribution before SMOTE:")
print(y_train.value_counts().sort_index())

print("\nTesting distribution (left unchanged):")
print(y_test.value_counts().sort_index())


# =========================================================
# 5. APPLY SMOTE TO TRAINING DATA ONLY
# =========================================================

smote = SMOTE(random_state=RANDOM_STATE)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining distribution after SMOTE:")
print(y_train_smote.value_counts().sort_index())

print(
    "\nThe test set has NOT been resampled. "
    "All experiments use the same test records."
)


# =========================================================
# 6. DEFINE MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# =========================================================
# 7. EVALUATION FUNCTION
# =========================================================

results = []


def evaluate_model(
    model,
    model_name,
    X_train_data,
    y_train_data,
    X_test_data,
    y_test_data,
    experiment_name
):
    """
    Train and evaluate one model.

    ROC-AUC uses the predicted probability for class 1,
    where class 1 represents machine failure.
    """

    print("\n" + "=" * 65)
    print(f"{model_name} | {experiment_name}")
    print("=" * 65)

    model.fit(X_train_data, y_train_data)

    y_pred = model.predict(X_test_data)

    # Locate the probability column corresponding to class 1.
    class_list = list(model.classes_)

    if 1 not in class_list:
        raise ValueError(
            f"{model_name} does not contain failure class 1."
        )

    failure_index = class_list.index(1)

    y_prob = model.predict_proba(X_test_data)[:, failure_index]

    accuracy = accuracy_score(y_test_data, y_pred)

    precision = precision_score(
        y_test_data,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test_data,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test_data,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_test_data, y_prob)

    cm = confusion_matrix(
        y_test_data,
        y_pred,
        labels=[0, 1]
    )

    print(f"\nAccuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_data,
            y_pred,
            target_names=["Healthy", "Machine Failure"],
            zero_division=0
        )
    )

    # Save a separate confusion matrix image for each experiment.
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    experiment_suffix = (
        experiment_name.lower()
        .replace(" ", "_")
    )

    cm_filename = (
        f"confusion_matrix_{safe_name}_{experiment_suffix}.png"
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Failure"],
        yticklabels=["Healthy", "Failure"]
    )

    plt.title(f"{model_name} - {experiment_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, cm_filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Save numerical results.
    results.append({
        "Model": model_name,
        "Experiment": experiment_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-score": round(f1 * 100, 2),
        "ROC-AUC": round(roc_auc * 100, 2),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1])
    })


# =========================================================
# 8. RUN BOTH EXPERIMENTS FOR ALL MODELS
# =========================================================

for model_name, model in models.items():

    # -----------------------------------------------------
    # Experiment A: Original imbalanced training data
    # -----------------------------------------------------

    # Use class_weight='balanced' for Random Forest and
    # Logistic Regression. Decision Tree uses its default
    # weighting in this baseline experiment.
    #
    # Each model is cloned/recreated for the next experiment.

    if model_name == "Random Forest":
        baseline_model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "Logistic Regression":
        baseline_model = SklearnPipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ])

    else:
        baseline_model = DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )

    evaluate_model(
        baseline_model,
        model_name,
        X_train,
        y_train,
        X_test,
        y_test,
        "Original Training Data"
    )

    # -----------------------------------------------------
    # Experiment B: SMOTE-balanced training data
    # -----------------------------------------------------

    # No class_weight='balanced' is used here.
    # This lets us evaluate SMOTE without combining it
    # with class weighting.

    if model_name == "Random Forest":
        smote_model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "Logistic Regression":
        smote_model = SklearnPipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        ])

    else:
        smote_model = DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )

    evaluate_model(
        smote_model,
        model_name,
        X_train_smote,
        y_train_smote,
        X_test,
        y_test,
        "SMOTE"
    )


# =========================================================
# 9. SAVE EVALUATION RESULTS
# =========================================================

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "model_evaluation_results.csv"
)

results_df.to_csv(csv_path, index=False)

# =========================================================
# FINAL MODEL COMPARISON
# =========================================================

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

# Convert the list of results into a Pandas DataFrame
results_df = pd.DataFrame(results)

# Display the comparison table
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save the comparison results
os.makedirs("evaluation_results", exist_ok=True)

csv_path = os.path.join(
    "evaluation_results",
    "model_evaluation_results.csv"
)

results_df.to_csv(csv_path, index=False)

print(f"\nComparison results saved to: {csv_path}")

# =========================================================
# 10. GENERATE MODEL COMPARISON GRAPH
# =========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

plot_df = results_df.melt(
    id_vars=["Model", "Experiment"],
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plot_df["Model and Experiment"] = (
    plot_df["Model"] + "\n" + plot_df["Experiment"]
)

plt.figure(figsize=(16, 7))

sns.barplot(
    data=plot_df,
    x="Model and Experiment",
    y="Score",
    hue="Metric"
)

plt.title(
    "Model Comparison: Original Data vs SMOTE",
    fontsize=15
)

plt.xlabel("Model and Training Method")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

plt.xticks(rotation=20, ha="right")
plt.legend(title="Evaluation Metric")

plt.tight_layout()

graph_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_smote.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nComparison graph saved to: {graph_path}")

print("\nAll experiments completed successfully.")