import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Scale features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)

# Define base classifiers
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Train classifiers
mlp.fit(X_train, y_train)
extra_trees.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# Define weights for each classifier
weights = [0.8, 0.1, 0.1]  # Adjusted weights for ensemble

# Using soft voting with correct classifier names
voting_clf_soft = VotingClassifier(
    estimators=[('mlp', mlp), ('extra_trees', extra_trees), ('lgbm', lgbm)],
    voting='soft',
    weights=weights  # Assign weights properly
)

# Train the soft voting classifier
voting_clf_soft.fit(X_train, y_train)

# Predict using soft voting
y_pred_soft = voting_clf_soft.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred_soft)
print(f"Weighted Soft Voting Accuracy: {accuracy:.4f}")
