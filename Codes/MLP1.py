import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB2_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Scale features for Neural Network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize results storage
results = []

# Iterate through the feature range from 20 to 52
for k in range(20, 53):
    try:
        # Perform ANOVA feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        
        # Initialize Neural Network model with random seed
        model = MLPClassifier(random_state=RANDOM_SEED, max_iter=500)
        
        # Perform 10-fold stratified cross-validation with random seed
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X_selected, y, cv=skf, scoring='accuracy')
        
        # Store results
        results.append({
            'num_features': k,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'selected_features': selected_features.tolist()
        })

    except Exception as e:
        print(f"Error for {k} features: {e}")

# Find the best feature set
best_result = max(results, key=lambda x: x['mean_accuracy'])

# Print best result
print("\nBest Feature Selection Results:")
print(f"Number of Features: {best_result['num_features']}")
print(f"Mean Accuracy: {best_result['mean_accuracy']:.4f}")
print(f"Standard Deviation: {best_result['std_accuracy']:.4f}")
print(f"Selected Features: {best_result['selected_features']}")

# Plot accuracy vs number of features
plt.figure(figsize=(10, 6))
plt.plot([result['num_features'] for result in results],
         [result['mean_accuracy'] for result in results],
         marker='o', label='Mean Accuracy')
plt.fill_between(
    [result['num_features'] for result in results],
    [result['mean_accuracy'] - result['std_accuracy'] for result in results],
    [result['mean_accuracy'] + result['std_accuracy'] for result in results],
    alpha=0.2, label='± 1 Std Dev'
)
plt.title('Model Accuracy vs Number of Selected Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.grid()
plt.show()
