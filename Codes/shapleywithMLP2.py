import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB3_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Check if dataset has the required columns
if "label" not in df.columns or "file_name" not in df.columns:
    raise ValueError("Dataset must contain 'label' and 'file_name' columns")

# Prepare data
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]
feature_names = X.columns

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train initial MLP Model for SHAP analysis
print("Training initial MLP model...")
initial_mlp = MLPClassifier( max_iter=500, random_state=42)
initial_mlp.fit(X_scaled, y)

# SHAP Analysis
print("Calculating SHAP values...")
n_background = 100
background_scaled = shap.kmeans(X_scaled, n_background)
explainer_mlp = shap.KernelExplainer(lambda x: initial_mlp.predict_proba(x)[:, 1], background_scaled)

# Calculate SHAP values for a subset of test data
n_samples_shap = 50
X_shap_scaled = X_scaled[:n_samples_shap]
shap_values_mlp = explainer_mlp.shap_values(X_shap_scaled)

# Calculate feature importance
feature_importance = np.abs(shap_values_mlp).mean(0)
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Select top features and compute accuracy for different feature set sizes
feature_range = range(20, 53)
accuracies = []
optimal_features = None
best_accuracy = 0

for n_features in feature_range:
    selected_features = [feature[0] for feature in sorted_features[:n_features]]
    X_selected = X[selected_features]
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    # 10-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = MLPClassifier(max_iter=500, random_state=42)
    cv_scores = cross_val_score(model, X_selected_scaled, y, cv=skf)
    mean_accuracy = cv_scores.mean()
    accuracies.append(mean_accuracy)
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        optimal_features = n_features

# Print the best number of features and accuracy
print(f"\nOptimal number of features: {optimal_features}")
print(f"Best accuracy: {best_accuracy:.4f}")

# Plot accuracy vs. number of features
plt.figure(figsize=(10, 6))
plt.plot(feature_range, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.axvline(optimal_features, color='r', linestyle='--', label=f'Optimal Features ({optimal_features})')
plt.xlabel('Number of Selected Features')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Feature Selection Performance')
plt.legend()
plt.grid(True)
plt.show()
