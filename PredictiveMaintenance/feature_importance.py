# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE ANALYSIS
# Predictive Maintenance System
# =========================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD TRAINED MODEL
# ---------------------------------------------------------

MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Check model/model.pkl"
    )

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# 2. DEFINE FEATURE NAMES
# ---------------------------------------------------------

features = [
    "Type",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# ---------------------------------------------------------
# 3. EXTRACT FEATURE IMPORTANCE
# ---------------------------------------------------------

# Confirm that the saved model supports feature importance.
if not hasattr(model, "feature_importances_"):
    raise TypeError(
        "The loaded model does not support feature importance. "
        "Check that model/model.pkl contains a Random Forest model."
    )

importance = model.feature_importances_

if len(importance) != len(features):
    raise ValueError(
        "The number of model features does not match "
        "the five configured feature names."
    )

# ---------------------------------------------------------
# 4. CREATE RESULTS TABLE
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==============================================")

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance": lambda value: f"{value:.4f}"
        }
    )
)

# Save results to CSV.
importance_df.to_csv(
    "feature_importance_results.csv",
    index=False
)

print("\nResults saved to feature_importance_results.csv")

# ---------------------------------------------------------
# 5. GENERATE BAR GRAPH
# ---------------------------------------------------------

plot_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["Feature"],
    plot_df["Importance"]
)

plt.xlabel("Feature Importance Score")
plt.ylabel("Machine Parameters")

plt.title(
    "Random Forest Feature Importance "
    "for Predictive Maintenance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

print("Graph saved to feature_importance.png")

plt.show()

# ---------------------------------------------------------
# 6. DISPLAY MOST IMPORTANT FEATURE
# ---------------------------------------------------------

top_feature = importance_df.iloc[0]

print("\n==============================================")
print("MOST IMPORTANT FEATURE")
print("==============================================")

print("Feature:", top_feature["Feature"])
print(
    "Importance:",
    round(top_feature["Importance"], 4)
)

print("\nFeature importance analysis completed.")