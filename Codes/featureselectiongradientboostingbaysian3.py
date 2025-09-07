# File path: scripts/model_optimization.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Global results tracker
global_results = []

# Define the objective function for Bayesian Optimization
def objective_function(
    n_features=10,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
):
    # Convert parameters to integers where needed
    n_features = int(n_features)
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    # Validate n_features
    n_features = max(5, min(X.shape[1], n_features))

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, test_idx in skf.split(X, y):
        # Split data into training and test folds
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Feature selection inside the fold
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        # Create and train Gradient Boosting Classifier
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        gb.fit(X_train_scaled, y_train)

        # Evaluate using accuracy
        y_pred = gb.predict(X_test_scaled)
        fold_accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(fold_accuracy)

    # Calculate the mean CV accuracy
    mean_cv_accuracy = np.mean(cv_scores)

    # Store results
    result = {
        "n_features": n_features,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "cv_accuracy": mean_cv_accuracy,
    }
    global_results.append(result)

    print(f"\nIteration Results:")
    for key, value in result.items():
        print(f"{key}: {value}")

    return mean_cv_accuracy


# Define parameter bounds for Bayesian Optimization
pbounds = {
    "n_features": (5, X.shape[1]),
    "n_estimators": (50, 300),
    "learning_rate": (0.01, 0.3),
    "max_depth": (3, 10),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
}

# Create Bayesian Optimization object
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42,
)

# Perform optimization
optimizer.maximize(init_points=10, n_iter=50)

# Sort global results
sorted_results = sorted(global_results, key=lambda x: x["cv_accuracy"], reverse=True)

print("\n\n--- Top 5 Best Configurations ---")
for result in sorted_results[:5]:
    print("\nConfiguration:")
    for key, value in result.items():
        print(f"{key}: {value}")
