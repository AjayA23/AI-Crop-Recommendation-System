<<<<<<< HEAD
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")
print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)
print("\nColumns:")
print(data.columns)

# Input features
X = data.drop("label", axis=1)

# Target
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create AI Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "model/crop_model.pkl")

print("\nModel saved successfully!")
=======
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")
print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)
print("\nColumns:")
print(data.columns)

# Input features
X = data.drop("label", axis=1)

# Target
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create AI Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "model/crop_model.pkl")

print("\nModel saved successfully!")
>>>>>>> 443212299466bb189c153eb7455166ec17e7417c
print("Location: model/crop_model.pkl")