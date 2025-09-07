import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from pycaret.classification import *
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set a global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Function to perform feature selection and model comparison
def select_and_compare_features(df, min_features=26, max_features=44):
    # Extract features and labels
    X = df.drop(["label", "file_name"], axis=1)
    y = df["label"]
    
    # Store results
    results = []
    
    # Iterate through feature count from min to max
    for num_features in range(min_features, max_features + 1):
        try:
            # Perform ANOVA feature selection
            selector = SelectKBest(score_func=f_classif, k=min(num_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support(indices=True)]
            
            # Prepare data for PyCaret
            data_selected = pd.concat([
                pd.DataFrame(X_selected, columns=selected_features), 
                y.reset_index(drop=True)
            ], axis=1)
            
            # Setup PyCaret with a fixed random seed
            clf1 = setup(data=data_selected, 
                         target='label', 
                         use_gpu=True, 
                         fold=10, 
                         session_id=RANDOM_SEED)
            
            # Compare models and get the best model
            print(f"Processing {num_features} features...")
            best = compare_models()
            
            # Generate plots for the best model
            plot_model(best, plot='learning')
            plot_model(best, plot='auc')
            plot_model(best, plot='confusion_matrix')
            
            # Save the best model pipeline
            save_model(best, f'my_best_pipeline_{num_features}_features')
            
            # Get model performance
            performance = pull()
            
            # Find the row for the best model
            best_model_row = performance[performance['Model'] == best.__class__.__name__]
            
            # Check if the row exists
            if not best_model_row.empty:
                results.append({
                    'num_features': num_features,
                    'best_model': best.__class__.__name__,
                    'accuracy': best_model_row['Accuracy'].values[0]
                })
            else:
                print(f"No performance data found for {best.__class__.__name__} with {num_features} features")
        
        except Exception as e:
            print(f"Error processing {num_features} features: {e}")
    
    # Check if results are empty
    if not results:
        print("No valid results found. Check your data and feature selection.")
        return None
    
    # Find the best result
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("\nBest Feature Selection Results:")
    print(f"Number of Features: {best_result['num_features']}")
    print(f"Best Model: {best_result['best_model']}")
    print(f"Accuracy: {best_result['accuracy']}")
    
    return results

# Load dataset
df = pd.read_csv('D:\new researches\Car research2\features csv files\DB3_Categorized_features1.csv')

# Perform feature selection and model comparison
feature_results = select_and_compare_features(df)

# Visualize results if possible
if feature_results:
    plt.figure(figsize=(10, 6))
    plt.plot([result['num_features'] for result in feature_results], 
             [result['accuracy'] for result in feature_results], 
             marker='o')
    plt.title('Model Accuracy vs Number of Selected Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
