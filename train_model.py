# train_model.py
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = "StressLevelDataset.csv"   # put the CSV next to this file

df = pd.read_csv(CSV_PATH)

target = "stress_level"
X = df.drop(columns=[target])
y = df[target]

# Keep the feature order for the app
feature_order = X.columns.tolist()

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Strong baseline model for tabular data
rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("=== Test metrics ===")
print(classification_report(y_test, rf.predict(X_test)))

# Save artifacts
joblib.dump(rf, "model_rf.joblib")
with open("feature_order.json", "w") as f:
    json.dump({"features": feature_order}, f, indent=2)

print("Saved: model_rf.joblib, feature_order.json")
