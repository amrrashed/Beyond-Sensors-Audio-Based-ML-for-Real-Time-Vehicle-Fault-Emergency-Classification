import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Ensure dataset has required columns
if "label" not in df.columns or "file_name" not in df.columns:
    raise ValueError("Dataset must contain 'label' and 'file_name' columns")

# Prepare data
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]
feature_names = X.columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an initial MLP model
print("Training initial MLP model...")
mlp = MLPClassifier(max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Compute SHAP values
print("Calculating SHAP values...")
background = shap.kmeans(X_train_scaled, 100)
explainer = shap.KernelExplainer(lambda x: mlp.predict_proba(x)[:, 1], background)
shap_values = explainer.shap_values(X_test_scaled[:50])  # Compute SHAP for a subset

# Rank features by importance
feature_importance = np.abs(shap_values).mean(axis=0)
sorted_features = np.argsort(feature_importance)[::-1]  # Sort in descending order
sorted_importance = feature_importance[sorted_features]
selected_features = [feature_names[i] for i in sorted_features]

# **Determine Optimal Number of Features Using Cumulative SHAP Contribution**
cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
optimal_n_features = np.argmax(cumulative_importance >= 0.95) + 1  # Select features covering 95% of total SHAP importance

print(f"Optimal number of features selected: {optimal_n_features}")

# Select top features based on SHAP importance
top_features = selected_features[:optimal_n_features]

# Train the final model with selected features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

mlp_final = MLPClassifier(max_iter=500, random_state=42)
mlp_final.fit(X_train_selected_scaled, y_train)

# Compute accuracy
final_accuracy = accuracy_score(y_test, mlp_final.predict(X_test_selected_scaled))
print(f"Final accuracy with {optimal_n_features} features: {final_accuracy:.4f}")

# Plot Cumulative SHAP Importance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(sorted_importance) + 1), cumulative_importance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=optimal_n_features, color='g', linestyle='--')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative SHAP Importance')
plt.title('Optimal Feature Selection Based on SHAP Importance')
plt.show()

# Plot Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(top_features[::-1], sorted_importance[:optimal_n_features][::-1])
plt.xlabel('Mean |SHAP Value|')
plt.ylabel('Feature')
plt.title(f'Top {optimal_n_features} Features Based on SHAP')
plt.show()
