# File path: scripts/feature_selection_models_kfold.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB3_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Encode categorical target labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check the mapping of labels to numerical values
print("Label Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")

# Initialize machine learning models
models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    "LightGBM": LGBMClassifier(random_state=RANDOM_SEED),
    "Neural Network": MLPClassifier(random_state=RANDOM_SEED, max_iter=500),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric="logloss"),
}

# Initialize results dictionary
results = {model_name: [] for model_name in models.keys()}

# Range of features to test
feature_range = list(range(15, 53))

# Perform standard K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Rank features using ANOVA
f_scores, _ = f_classif(X, y_encoded)
ranked_features = np.argsort(f_scores)[::-1]  # Indices of features sorted by importance

# Loop over the number of features
for n_features in feature_range:
    print(f"Testing with top {n_features} features...")

    # Select top-n features
    selected_features = X.columns[ranked_features[:n_features]]
    X_selected = X[selected_features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Cross-validation for each model
    for model_name, model in models.items():
        cv_scores = []

        for train_idx, test_idx in kf.split(X_scaled):
            # Train/Test split for each fold
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores.append(accuracy)

        # Store the average accuracy
        mean_accuracy = np.mean(cv_scores)
        results[model_name].append(mean_accuracy)

# Plot results
plt.figure(figsize=(12, 8))

for model_name, accuracies in results.items():
    plt.plot(feature_range, accuracies, label=model_name)

plt.title("Accuracy vs Number of Selected Features (K-Fold Cross-Validation)")
plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
