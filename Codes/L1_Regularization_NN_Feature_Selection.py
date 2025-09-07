import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K

# Load the dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Debug: Print dataset shape
print(f"Original dataset shape: Features = {X.shape[1]}, Samples = {X.shape[0]}")

# Encode the target variable if it's categorical
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train is numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Debug: Verify no NaN or non-numeric values
print(f"Post-conversion X_train shape: {X_train.shape}")
print(f"Post-conversion X_test shape: {X_test.shape}")

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Debug: Verify standardized shapes
print(f"Standardized X_train shape: {X_train.shape}")
print(f"Standardized X_test shape: {X_test.shape}")

# Define the neural network model with L1 regularization
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input shape matches the number of features
    Dense(64, activation='relu', kernel_regularizer='l1'),
    Dense(32, activation='relu', kernel_regularizer='l1'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Retrieve the weights of the first dense layer
first_layer_weights = model.layers[0].get_weights()[0]  # Weights of the input layer
weights_abs_sum = np.sum(np.abs(first_layer_weights), axis=1)  # Sum absolute weights for each input feature

# Debug: Check feature-weight alignment
print(f"Number of features: {X.shape[1]}, Number of weights: {len(weights_abs_sum)}")
if len(weights_abs_sum) != X.shape[1]:
    raise ValueError("Mismatch between model weights and feature columns. Ensure correct preprocessing.")

# Rank features based on their importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": weights_abs_sum
})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Save the ranked features to a CSV file
output_path = r"D:\new researches\Car research2\features csv files\DB1_ranked_features.csv"
feature_importance.to_csv(output_path, index=False)
print(f"Ranked features saved to {output_path}")

# Display the top features
print("Top ranked features:")
print(feature_importance.head())
