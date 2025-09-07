import numpy as np
from bayes_opt import BayesianOptimization
from scipy.optimize import Bounds  # If using scipy
from skopt.space import Real  # Importing Real from scikit-optimize (skopt)
from evaluate_weighted_voting import evaluate_weighted_voting  
import warnings
warnings.filterwarnings("ignore")
 

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define parameter bounds for Bayesian Optimization
weights_bounds = {
    'w1': (0, 1),
    'w2': (0, 1),
    'w3': (0, 1),
}

# File path for dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"

# Define an objective function wrapper for Bayesian Optimization
def objective(w1, w2, w3):
    weights = [w1, w2, w3]  # Ensure correct weight format
    return evaluate_weighted_voting(data_path, weights)  # Ensure correct function usage

# Create Bayesian Optimization object
optimizer = BayesianOptimization(
    f=objective,
    pbounds=weights_bounds,
    random_state=42
)

# Perform optimization
optimizer.maximize(
    init_points=10,  # Number of initial random points
    n_iter=50        # Number of iterations
)

# Optimize Hard Voting
best_params = optimizer.max['params']  # Extract best weights
best_weights = [best_params['w1'], best_params['w2'], best_params['w3']]
accuracy = evaluate_weighted_voting(data_path, best_weights)

print(f"Best Weights: {best_weights}, Accuracy: {accuracy}")



