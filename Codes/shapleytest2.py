import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# Load and split the dataset
print("Loading and preparing data...")
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train initial model and get SHAP values
print("Training initial model and calculating SHAP values...")
initial_model = xgb.XGBClassifier(random_state=42)
initial_model.fit(X_train, y_train)
explainer = shap.Explainer(initial_model)
shap_values = explainer(X_train)

# Calculate feature importance
feature_importance = np.abs(shap_values.values).mean(0)
feature_importance_dict = dict(zip(X.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

def objective_function(n_features):
    # Convert n_features to integer
    n_features = int(n_features)
    
    # Select top n features
    selected_features = [feature[0] for feature in sorted_features[:n_features]]
    
    # Create dataset with selected features
    X_train_selected = X_train[selected_features]
    
    # Train model and get cross-validation score
    model = xgb.XGBClassifier(random_state=42)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
    
    return cv_scores.mean()

# Define optimization bounds
n_features_min = max(3, len(sorted_features) // 10)  # At least 3 features or 10% of total
n_features_max = len(sorted_features)

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds={'n_features': (n_features_min, n_features_max)},
    random_state=42
)

# Run optimization
print("\nRunning Bayesian optimization...")
optimizer.maximize(
    init_points=5,  # Number of random initial points
    n_iter=15,      # Number of optimization steps
)

# Get optimal number of features
optimal_n_features = int(optimizer.max['params']['n_features'])
print(f"\nOptimal number of features: {optimal_n_features}")

# Train final model with optimal number of features
print("\nTraining final model with optimal features...")
selected_features = [feature[0] for feature in sorted_features[:optimal_n_features]]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

final_model = xgb.XGBClassifier(random_state=42)
final_model.fit(X_train_selected, y_train)

# Calculate accuracies
initial_accuracy = accuracy_score(y_test, initial_model.predict(X_test))
final_accuracy = accuracy_score(y_test, final_model.predict(X_test_selected))

print(f"\nResults:")
print(f"Initial accuracy (all features): {initial_accuracy:.4f}")
print(f"Final accuracy (optimal features): {final_accuracy:.4f}")
print(f"Accuracy difference: {final_accuracy - initial_accuracy:.4f}")
print(f"\nSelected features:")
for feature in selected_features:
    print(f"- {feature}")

# Plot optimization history
plt.figure(figsize=(10, 6))
optimization_history = [res['target'] for res in optimizer.res]
plt.plot(range(len(optimization_history)), optimization_history, 'b-', label='Optimization progress')
plt.scatter(range(len(optimization_history)), optimization_history, c='b')
plt.xlabel('Iteration')
plt.ylabel('Cross-validation Accuracy')
plt.title('Bayesian Optimization Progress')
plt.legend()
plt.grid(True)
plt.show()

# Plot feature importance for selected features
plt.figure(figsize=(12, 6))
selected_importance = [importance for feature, importance in sorted_features[:optimal_n_features]]
plt.bar(range(optimal_n_features), selected_importance)
plt.xticks(range(optimal_n_features), selected_features, rotation=45, ha='right')
plt.xlabel('Selected Features')
plt.ylabel('Mean |SHAP value|')
plt.title('Feature Importance of Selected Features')
plt.tight_layout()
plt.show()

# Create SHAP summary plot for selected features
shap_values_selected = explainer(X_test_selected)
shap.summary_plot(shap_values_selected, X_test_selected)

# Print optimization details
print("\nOptimization Details:")
print("Best score:", optimizer.max['target'])
print("Best parameters:", optimizer.max['params'])
print("\nOptimization History:")
for i, res in enumerate(optimizer.res):
    print(f"Iteration {i}: Score = {res['target']:.4f}, n_features = {int(res['params']['n_features'])}")