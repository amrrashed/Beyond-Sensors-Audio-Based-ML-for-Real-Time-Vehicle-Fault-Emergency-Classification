import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
try:
    data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
    df = pd.read_csv(data_path)
    print("Data shape:", df.shape)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Prepare data
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]
feature_names = X.columns.tolist()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DMatrix for better performance
dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)

# Train XGBoost model
params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y)),
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42
}

xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Make predictions
y_pred = xgb_model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# SHAP Analysis
try:
    # Create explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_scaled)
    
    # If shap_values is a list (multi-class), take the mean across classes
    if isinstance(shap_values, list):
        shap_values = np.mean(shap_values, axis=0)
    
    # Feature importance based on SHAP
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plotting
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values if not isinstance(shap_values, list) else shap_values[0],
        X_test_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error in SHAP analysis: {e}")
    raise