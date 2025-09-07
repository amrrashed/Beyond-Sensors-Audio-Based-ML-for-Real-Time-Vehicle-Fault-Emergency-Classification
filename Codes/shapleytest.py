import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load a standard dataset
X, y = shap.datasets.adult()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train an XGBoost classifier
model = xgb.XGBClassifier().fit(X_train, y_train)

# Initialize the SHAP Explainer
explainer = shap.Explainer(model)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# Visualize the SHAP values for the first prediction in the test set
shap.plots.waterfall(shap_values[0], max_display=14)