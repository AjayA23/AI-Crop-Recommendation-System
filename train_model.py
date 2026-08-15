import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= LOAD DATASET =================

data = pd.read_csv("Crop_recommendation.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

print("\nColumns:")
print(data.columns)


# ================= INPUT & TARGET =================

X = data.drop("label", axis=1)

y = data["label"]


# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================= CREATE AI MODEL =================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ================= TRAIN MODEL =================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ================= PREDICTION =================

predictions = model.predict(
    X_test
)


# ================= ACCURACY =================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nModel Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ================= CREATE MODEL FOLDER =================

os.makedirs(
    "model",
    exist_ok=True
)


# ================= SAVE MODEL =================

model_path = "model/crop_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully!")
print("Location:", model_path)