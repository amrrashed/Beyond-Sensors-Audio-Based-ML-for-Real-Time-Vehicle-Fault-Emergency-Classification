#installed packages 
#pip install pandas numpy scikit-learn scikit-optimize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from skopt import BayesSearchCV
from skopt.space import Integer, Real
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

# Select top features with Bayesian optimization
def select_features_and_train(n_features):
    # Select top n features
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
    
    # Define parameter space for Bayesian optimization
    param_space = {
        'n_estimators': Integer(50, 300),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'max_depth': Integer(3, 10),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10)
    }
    
    # Create Gradient Boosting Classifier
    gb = GradientBoostingClassifier(random_state=42)
    
    # Bayesian optimization with cross-validation
    opt_search = BayesSearchCV(
        gb, 
        param_space, 
        n_iter=50, 
        cv=5, 
        scoring='accuracy', 
        random_state=42
    )
    
    # Fit the optimized search
    opt_search.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    best_model = opt_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    print(f"\nNumber of features: {n_features}")
    print("Best Parameters:", opt_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return opt_search.best_score_

# Iterate through feature selections from 5 to 52
results = []
for n in range(5, 53):
    score = select_features_and_train(n)
    results.append((n, score))

# Print overall results
print("\nFeature Selection Performance:")
for n, score in results:
    print(f"Features: {n}, Best CV Score: {score}")