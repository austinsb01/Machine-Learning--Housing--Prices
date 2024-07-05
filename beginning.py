# beginning.py

from utils import load_data, compute_cost, compute_gradient, gradient_descent, preprocess_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load training data
X, y = load_data("house-prices-advanced-regression-techniques/train.csv")

# Pick apart the numeric and categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# One Hot Encoder: Takes all categorical values and makes more categories according to their possible values
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical_features = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=encoded_feature_names)
data_preprocessed = pd.concat([X[numeric_features], encoded_categorical_df], axis=1)
X = data_preprocessed.values  # Feature matrix

# Handle missing values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize y
y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / y_std

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize weights and bias
m, n = X_train_scaled.shape
initial_w = np.zeros(n)
initial_b = 0.0

# Print number of features
print(f'Number of features: {n}')

# Compute the cost with initial weights and bias
initial_cost = compute_cost(X_train_scaled, y_train, initial_w, initial_b)
print(f'Initial cost at initial w and b: {initial_cost}')

# Compute gradient
dj_db, dj_dw = compute_gradient(X_train_scaled, y_train, initial_w, initial_b)

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(n) - 0.5)
initial_b = -8

# Gradient descent settings
iterations = 10000
alpha = 0.001

# Train the model using gradient descent
w, b, J_history, w_history = gradient_descent(X_train_scaled, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

# Print cost at the 9999th iteration
print(f'Scaled cost at the 9999th iteration: {J_history[9999]}')

# Make predictions on the test set
predictions = np.dot(X_test_scaled, w) + b

# Convert predictions back to the original scale
predictions_original_scale = predictions * y_std + y_mean
y_test_original_scale = y_test * y_std + y_mean

# Evaluate the model
mae = mean_absolute_error(y_test_original_scale, predictions_original_scale)
mse = mean_squared_error(y_test_original_scale, predictions_original_scale)
r2 = r2_score(y_test_original_scale, predictions_original_scale)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

# Calculate percentage accuracy
percentage_within_10_percent = np.mean(np.abs((y_test_original_scale - predictions_original_scale) / y_test_original_scale) <= 0.10) * 100
print(f'Percentage of predictions within 10% of actual values: {percentage_within_10_percent:.2f}%')



#when interpreting results, run the following to get accurate values of y:
#y_pred = some_model_predict_function(X_test)
#y_pred_original_scale = y_pred * y_std + y_mean