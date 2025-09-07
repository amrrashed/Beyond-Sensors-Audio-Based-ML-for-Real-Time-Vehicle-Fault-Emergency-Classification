import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB3_Categorized_features1.csv"
df = pd.read_csv(data_path)
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

feature_names = X.columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Cross-validation score
cv_scores_mlp = cross_val_score(mlp, X_train_scaled, y_train, cv=10)
print(f"MLP Cross-Validation Score: {cv_scores_mlp.mean():.4f}")

# SHAP Analysis
n_samples_shap = 50
n_background = 100
background_scaled = shap.kmeans(X_train_scaled, n_background)
X_test_shap_scaled = X_test_scaled[:n_samples_shap]

explainer_mlp = shap.KernelExplainer(lambda x: mlp.predict_proba(x)[:, 1], background_scaled)
print("Calculating SHAP values (this may take a few minutes)...")
shap_values_mlp = explainer_mlp.shap_values(X_test_shap_scaled)

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'MLP': np.abs(shap_values_mlp).mean(axis=0)
}).sort_values('MLP', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Visualization
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values_mlp, X_test_shap_scaled, feature_names=feature_names, plot_type="bar")
plt.title("MLP SHAP Feature Importance")
plt.show()
