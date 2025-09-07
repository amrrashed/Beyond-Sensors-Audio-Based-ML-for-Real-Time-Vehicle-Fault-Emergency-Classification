#installed packages
#pip install pandas numpy scikit-learn bayesian-optimization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from bayes_opt import BayesianOptimization
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(['label', 'file_name'], axis=1)
y = df['label']

# Perform ANOVA F-test for feature ranking
f_scores, _ = f_classif(X, y)
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': f_scores
})
ranked_features = feature_scores.sort_values('f_score', ascending=False)

# Global results tracker
global_results = []

def objective_function(n_features=10, n_estimators=100, learning_rate=0.1, 
                       max_depth=5, min_samples_split=10, min_samples_leaf=5):
    # Convert parameters to integers where needed
    n_features = int(n_features)
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    
    # Validate n_features
    n_features = max(5, min(52, n_features))
    
    # Select top features
    top_features = ranked_features['feature'][:n_features].tolist()
    X_selected = X[top_features]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train Gradient Boosting Classifier
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(gb, X_train_scaled, y_train, cv=5)
    mean_cv_accuracy = np.mean(cv_scores)
    
    # Fit and evaluate on test set
    gb.fit(X_train_scaled, y_train)
    y_pred = gb.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Store and print results
    result = {
        'n_features': n_features,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'cv_accuracy': mean_cv_accuracy,
        'test_accuracy': test_accuracy
    }
    global_results.append(result)
    
    print(f"\nIteration Results:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    return mean_cv_accuracy

# Define parameter bounds for Bayesian Optimization
pbounds = {
    'n_features': (5, 52),
    'n_estimators': (50, 300),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# Create Bayesian Optimization object
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42
)

# Perform optimization
optimizer.maximize(
    init_points=10,  # Number of initial random points
    n_iter=50       # Number of iterations
)

# Sort global results
sorted_results = sorted(global_results, key=lambda x: x['cv_accuracy'], reverse=True)

print("\n\n--- Top 5 Best Configurations ---")
for result in sorted_results[:5]:
    print("\nConfiguration:")
    for key, value in result.items():
        print(f"{key}: {value}")