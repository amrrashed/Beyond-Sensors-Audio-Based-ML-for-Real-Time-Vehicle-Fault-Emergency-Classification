import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Scale features for Neural Network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define base classifiers
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Hard Voting Classifier
hard_voting = VotingClassifier(estimators=[('mlp', mlp), ('extra', extra_trees), ('lgbm', lgbm)], voting='hard')

# Soft Voting Classifier
soft_voting = VotingClassifier(estimators=[('mlp', mlp), ('extra', extra_trees), ('lgbm', lgbm)], voting='soft')

# Train classifiers
hard_voting.fit(X_train, y_train)
soft_voting.fit(X_train, y_train)

# Make predictions
y_pred_hard = hard_voting.predict(X_test)
y_pred_soft = soft_voting.predict(X_test)

# Evaluate performance
acc_hard = accuracy_score(y_test, y_pred_hard)
acc_soft = accuracy_score(y_test, y_pred_soft)

print(f'Hard Voting Accuracy: {acc_hard:.4f}')
print(f'Soft Voting Accuracy: {acc_soft:.4f}')
