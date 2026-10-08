import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

# Load Dataset

df = pd.read_csv(
    "data/synthetic_disaster_events_2025.csv"
)
#remove date column

df.drop('date', axis=1, inplace=True)

# Remove ID column
df.drop('event_id', axis=1, inplace=True)

# Encode categorical columns

le_disaster = LabelEncoder()
le_location = LabelEncoder()
le_aid = LabelEncoder()

df['disaster_type'] = le_disaster.fit_transform(
    df['disaster_type']
)

df['location'] = le_location.fit_transform(
    df['location']
)

df['aid_provided'] = le_aid.fit_transform(
    df['aid_provided']
)

# Feature Selection

selected_features = [
    'severity_level',
    'affected_population',
    'estimated_economic_loss_usd',
    'infrastructure_damage_index'
]
X = df[selected_features]

y = df['is_major_disaster']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

# Evaluation

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("Accuracy :", accuracy)
print("F1 Score :", f1)

# Save Model

joblib.dump(
    model,
    "models/disaster_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully!")