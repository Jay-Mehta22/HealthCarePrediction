import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- LOAD DATA ----------------
column_names = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv("heart.csv", header=None, names=column_names)

# Binarize target (UCI uses 0-4 scale)
df["target"] = (df["target"] > 0).astype(int)

print(f"✅ Rows before dedup: {len(df)}")

# FIX: remove duplicate rows to prevent data leakage
df = df.drop_duplicates()

print(f"✅ Rows after dedup:  {len(df)}")
print(f"✅ Target distribution:\n{df['target'].value_counts()}\n")

# ---------------- FEATURES ----------------
X = df.drop("target", axis=1)
y = df["target"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- SCALER ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------- MODELS TO COMPARE ----------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

# ---------------- CROSS VALIDATION ----------------
print("✅ Cross-Validation Results (5-Fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_cv_score = 0
best_name = ""

for name, m in models.items():
    cv_scores = cross_val_score(m, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    mean_score = cv_scores.mean()
    print(f"   {name}: {mean_score*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_model = m
        best_name = name

print(f"\n🏆 Best model: {best_name} ({best_cv_score*100:.2f}% CV accuracy)")

# ---------------- TRAIN BEST MODEL ----------------
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# ---------------- METRICS ----------------
test_acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (Random Forest / GBM only)
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\n✅ Feature Importance:")
    print(importance_df.to_string(index=False))

# ---------------- SAVE ----------------
joblib.dump(best_model, "heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")

print("\n✅ heart_model.pkl saved!")
print("✅ heart_scaler.pkl saved!")
print("\n🎉 Training complete — real accuracy, no data leakage.")
