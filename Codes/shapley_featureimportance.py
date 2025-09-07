import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
data_path = r"D:\new researches\Car research2\features csv files\DB3_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Drop unnecessary columns and separate features/labels
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Store feature names
feature_names = X.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled arrays back into DataFrames to keep feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# ----------------------------
# 2. Model Training
# ----------------------------
# Train an MLP classifier (you can adjust hyperparameters as needed)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# ----------------------------
# 3. SHAP Analysis for Feature Ranking
# ----------------------------
# Use a subset of the test set for SHAP to reduce computation time
n_samples_shap = 50
n_background = 100  # Number of clusters for background data

# Prepare background data using k-means clustering (recommended for KernelExplainer)
background_scaled = shap.kmeans(X_train_scaled, n_background)

# Use the first n_samples_shap samples from the test set for SHAP analysis
X_shap = X_test_scaled[:n_samples_shap]

# Create a SHAP KernelExplainer for the MLP model (using the probability of the positive class)
explainer = shap.KernelExplainer(lambda x: mlp.predict_proba(x)[:, 1], background_scaled)

print("Calculating SHAP values (this may take a few minutes)...")
shap_values = explainer.shap_values(X_shap)

# Optional: Display SHAP summary plots for visualization
plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    show=False
)
plt.title("MLP SHAP Summary Plot (Dot Plot)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("MLP SHAP Feature Importance (Bar Plot)")
plt.tight_layout()
plt.show()

# ----------------------------
# 4. Feature Ranking and CSV Saving
# ----------------------------
# Compute the mean absolute SHAP value for each feature (global importance)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_ABS_SHAP': np.abs(shap_values).mean(axis=0)
})

# Sort features by importance (highest to lowest)
feature_importance = feature_importance.sort_values(by='Mean_ABS_SHAP', ascending=False)
print("Top 52 most important features:")
print(feature_importance.head(52))

# Save the features by importance (highest to lowest)
feature_importance.to_csv('feature_importance_dataset3.csv', index=False)
print("New feature_importance saved as 'feature_importance_dataset3.csv'")

# Select the top 25 features
top_52_features = feature_importance.head(52)['Feature'].tolist()

# Create a new DataFrame containing only the top 25 features and the label
# (Here, we use the original dataframe 'df' to keep the original, unscaled values)
df_top52 = df[top_52_features + ['label']]

# Save the new dataset to a CSV file for future use
df_top52.to_csv('top_52_features_dataset3.csv', index=False)
print("New dataset with top 52 features saved as 'top_52_features_dataset3.csv'")
