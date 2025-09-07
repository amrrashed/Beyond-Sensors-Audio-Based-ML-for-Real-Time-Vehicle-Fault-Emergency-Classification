import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score

# Load and prepare dataset
data_path = r"D:\new researches\CAR RESEARCH\features csv files\DB3_features1.csv"
df = pd.read_csv(data_path)
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Store feature names
feature_names = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize features for all models (SGD requires scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame to maintain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# Models
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
sgd = SGDClassifier(
    loss='log_loss',  # for logistic regression loss
    max_iter=1000,
    random_state=42,
    learning_rate='adaptive',
    eta0=0.01,
    tol=1e-3
)

# Train Models
mlp.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)
sgd.fit(X_train_scaled, y_train)

# Cross-validation scores
cv_scores_mlp = cross_val_score(mlp, X_train_scaled, y_train, cv=10)
cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=10)
cv_scores_sgd = cross_val_score(sgd, X_train_scaled, y_train, cv=10)

print("Cross-Validation Scores:")
print(f"MLP: {cv_scores_mlp.mean():.4f}")
print(f"LR: {cv_scores_lr.mean():.4f}")
print(f"SGD: {cv_scores_sgd.mean():.4f}")

# SHAP Analysis
# Use smaller subset for SHAP analysis to reduce computation time
n_samples_shap = 50
n_background = 100

# Prepare background data
background_scaled = shap.kmeans(X_train_scaled, n_background)

# Prepare test samples for SHAP
X_test_shap_scaled = X_test_scaled[:n_samples_shap]

# SHAP Explainers
explainer_mlp = shap.KernelExplainer(
    lambda x: mlp.predict_proba(x)[:, 1],
    background_scaled
)

explainer_lr = shap.KernelExplainer(
    lambda x: lr.predict_proba(x)[:, 1],
    background_scaled
)

explainer_sgd = shap.KernelExplainer(
    lambda x: sgd.predict_proba(x)[:, 1],
    background_scaled
)

# Calculate SHAP values
print("Calculating SHAP values (this may take a few minutes)...")
shap_values_mlp = explainer_mlp.shap_values(X_test_shap_scaled)
shap_values_lr = explainer_lr.shap_values(X_test_shap_scaled)
shap_values_sgd = explainer_sgd.shap_values(X_test_shap_scaled)

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'MLP': np.abs(shap_values_mlp).mean(axis=0),
    'LR': np.abs(shap_values_lr).mean(axis=0),
    'SGD': np.abs(shap_values_sgd).mean(axis=0)
})

# Normalize feature importance
for model in ['MLP', 'LR', 'SGD']:
    feature_importance[f'{model}_normalized'] = feature_importance[model] / feature_importance[model].sum()

# Calculate model weights based on CV performance
weights = {
    'MLP': cv_scores_mlp.mean(),
    'LR': cv_scores_lr.mean(),
    'SGD': cv_scores_sgd.mean()
}
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

print("\nModel Weights:")
for model, weight in weights.items():
    print(f"{model}: {weight:.4f}")

# Make predictions on full test set
y_pred_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_sgd = sgd.predict_proba(X_test_scaled)[:, 1]

# Weighted ensemble predictions
y_pred_proba = (
    weights['MLP'] * y_pred_mlp +
    weights['LR'] * y_pred_lr +
    weights['SGD'] * y_pred_sgd
)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nEnsemble Model Accuracy: {accuracy:.4f}")

# Save top features and their importance scores
top_features = feature_importance.sort_values('MLP_normalized', ascending=False)
print("\nTop 10 Most Important Features (MLP):")
print(top_features[['Feature', 'MLP_normalized']].head(25))

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
    shap_values_sgd,
    X_test_shap_scaled,
    feature_names=feature_names
)
plt.title("SGD SHAP Feature Importance")
plt.tight_layout()
plt.show()

# Create force plot for first prediction
sample_idx = 0
plt.figure(figsize=(20, 3))
shap.force_plot(
    explainer_mlp.expected_value,
    shap_values_mlp[sample_idx],
    X_test_shap_scaled.iloc[sample_idx],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.title("SHAP Force Plot - First Sample")
plt.tight_layout()
plt.show()

# Save feature importance rankings to a CSV file
feature_importance.to_csv('feature_importance_rankings.csv', index=False)
print("\nFeature importance rankings have been saved to 'feature_importance_rankings.csv'")

