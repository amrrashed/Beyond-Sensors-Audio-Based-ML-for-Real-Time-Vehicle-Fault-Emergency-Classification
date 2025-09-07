import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from boruta import BorutaPy

# Load the dataset
data_path = r"D:\new researches\Car research2\features csv files\DB3_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Initialize an ExtraTreesClassifier for Boruta
et = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced', random_state=42)

# Initialize Boruta
boruta_selector = BorutaPy(estimator=et, n_estimators='auto', random_state=42)

# Fit Boruta on the dataset
boruta_selector.fit(X.values, y.values)

# Get selected features
selected_features = X.columns[boruta_selector.support_]
print("Selected Features:")
print(selected_features)

# Get tentative features
tentative_features = X.columns[boruta_selector.support_weak_]
print("Tentative Features:")
print(tentative_features)

# Reduce the dataset to only selected features
X_selected = X[selected_features]

# Add the label column back to the reduced dataset
X_selected["label"] = y

# Output dataset with selected features and labels
output_path = r"D:\new researches\Car research2\features csv files\DB3_selected_features_ExtraTrees.csv"
X_selected.to_csv(output_path, index=False)
print(f"Reduced dataset with labels saved to {output_path}")
