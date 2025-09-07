import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

feature_names = X.columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for MLP and LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train Models
mlp.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)
adaboost.fit(X_train, y_train)

# Cross-validation scores
cv_scores_mlp = cross_val_score(mlp, X_train_scaled, y_train, cv=10)
cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=10)
cv_scores_adaboost = cross_val_score(adaboost, X_train, y_train, cv=10)

print("Cross-Validation Scores:")
print(f"MLP: {cv_scores_mlp.mean():.4f}")
print(f"LR: {cv_scores_lr.mean():.4f}")
print(f"AdaBoost: {cv_scores_adaboost.mean():.4f}")

# SHAP Analysis
# Use smaller subset for SHAP analysis to reduce computation time
n_samples_shap = 50
n_background = 100

# Prepare background data
background_scaled = shap.kmeans(X_train_scaled, n_background)
background_unscaled = shap.kmeans(X_train, n_background)

# Prepare test samples for SHAP
X_test_shap_scaled = X_test_scaled[:n_samples_shap]
X_test_shap = X_test[:n_samples_shap]

# SHAP Explainers
explainer_mlp = shap.KernelExplainer(
    lambda x: mlp.predict_proba(x)[:, 1],
    background_scaled
)

explainer_lr = shap.KernelExplainer(
    lambda x: lr.predict_proba(x)[:, 1],
    background_scaled
)

explainer_adaboost = shap.KernelExplainer(
    lambda x: adaboost.predict_proba(x)[:, 1],
    background_unscaled
)

# Calculate SHAP values
print("Calculating SHAP values (this may take a few minutes)...")
shap_values_mlp = explainer_mlp.shap_values(X_test_shap_scaled)
shap_values_lr = explainer_lr.shap_values(X_test_shap_scaled)
shap_values_adaboost = explainer_adaboost.shap_values(X_test_shap)

# Calculate feature importance
num_features = X_train.shape[1]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'MLP': np.abs(shap_values_mlp).mean(axis=0),
    'LR': np.abs(shap_values_lr).mean(axis=0),
    'AdaBoost': np.abs(shap_values_adaboost).mean(axis=0)
})

# Normalize feature importance
for model in ['MLP', 'LR', 'AdaBoost']:
    feature_importance[f'{model}_normalized'] = feature_importance[model] / feature_importance[model].sum()

# Calculate model weights based on CV performance
weights = {
    'MLP': cv_scores_mlp.mean(),
    'LR': cv_scores_lr.mean(),
    'AdaBoost': cv_scores_adaboost.mean()
}
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

print("\nModel Weights:")
for model, weight in weights.items():
    print(f"{model}: {weight:.4f}")

# Make predictions on full test set
y_pred_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_adaboost = adaboost.predict_proba(X_test)[:, 1]

# Weighted ensemble predictions
y_pred_proba = (
    weights['MLP'] * y_pred_mlp +
    weights['LR'] * y_pred_lr +
    weights['AdaBoost'] * y_pred_adaboost
)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nEnsemble Model Accuracy: {accuracy:.4f}")

# Visualizations
plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values_mlp,
    X_test_shap_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("MLP SHAP Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values_lr,
    X_test_shap_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("Logistic Regression SHAP Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values_adaboost,
    X_test_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("AdaBoost SHAP Feature Importance")
plt.tight_layout()
plt.show()

# Force plot for first prediction
# Create force plot for first prediction
sample_idx = 0
plt.figure(figsize=(20, 3))
shap.force_plot(
    explainer_mlp.expected_value,
    shap_values_mlp[sample_idx],
    X_test_shap_scaled[sample_idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title("SHAP Force Plot - First Sample")
plt.tight_layout()
plt.show()

# Save feature importance to DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(shap_values_mlp).mean(0)
}).sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())