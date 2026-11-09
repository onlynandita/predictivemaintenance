import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


df = pd.read_csv("dataset/ai4i2020.csv")


df.drop(["UDI", "Product ID"], axis=1, inplace=True)


numeric_df = df.select_dtypes(include=["int64", "float64"])

corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


df.drop(["Air temperature [K]"], axis=1, inplace=True)

print("\nRemaining Columns:")
print(df.columns)


le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])


X = df[
    [
        "Type",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]


y = df["Machine failure"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(class_weight='balanced',random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nAccuracy:", accuracy_score(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


joblib.dump(model, "model/model.pkl")

print("\nModel saved successfully!")