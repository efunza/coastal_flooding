import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA = Path("data")
df = pd.read_csv(DATA / "features.csv")

FEATURES = ["rain_3day", "rain_7day", "wave_h", "elevation", "slope", "dist_coast"]
TARGET = "flooded"

X = df[FEATURES]
y = df[TARGET]

# Split by random rows for demo; for a real project, split by event/date
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred))

joblib.dump({"model": model, "features": FEATURES}, DATA / "model.joblib")
print("✅ Saved data/model.joblib")
