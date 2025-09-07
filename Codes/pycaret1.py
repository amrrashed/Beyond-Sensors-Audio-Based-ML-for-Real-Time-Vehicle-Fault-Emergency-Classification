# Load dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

# Load your dataset
df = pd.read_csv('DB2_Categorized_features1.csv')

# Extract features and labels
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]


# Plot class distribution histogram
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', data=df)  # Corrected 'class' to 'label'
plt.title('Class Distribution Histogram')
plt.xlabel('Class')
plt.ylabel('Count')

# Add numbers to the head of each column
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# Show dataset summary
dataset_summary = df.describe(include='all')

# Perform ANOVA feature selection
selector = SelectKBest(score_func=f_classif, k=34)  # Select top 
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support(indices=True)]

print("Top 38 selected features:")
print(selected_features)

# PyCaret Setup
from pycaret.classification import *

# Setup with selected features, GPU support, and 5-fold cross-validation
clf1 = setup(data=pd.concat([pd.DataFrame(X_selected, columns=selected_features), y.reset_index(drop=True)], axis=1), 
             target='label', 
             use_gpu=True, 
             fold=10)

# Compare models
best = compare_models()

# Evaluate best model
evaluate_model(best)
