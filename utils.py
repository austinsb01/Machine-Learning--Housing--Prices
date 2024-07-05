import pandas as pd
import numpy as np
import math

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]  # All cols except the last one
    y = data.iloc[:, -1]   # The last column
    return X, y

def compute_cost(X, y, w, b):
    m = len(y)
    z = np.dot(X, w) + b  
    cost = (1 / (2 * m)) * np.sum((z - y) ** 2)
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    z = np.dot(X, w) + b
    error = z - y

    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)
    
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    
    # The array to store cost J and w's at each iteration primarily for graphing later (copied from jupyter, idk if I need this)
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i < 100000:  #safeguard thta chatgpt taught me about
            cost = cost_function(X, y, w_in, b_in)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w_in, b_in, J_history, w_history


def preprocess_data(X, encoder, scaler, numeric_features, categorical_features):
    # Encode categorical features
    encoded_categorical_features = encoder.transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=encoded_feature_names)
    
    # Combine numeric and encoded categorical features
    data_preprocessed = pd.concat([X[numeric_features], encoded_categorical_df], axis=1)
    X = data_preprocessed.values  # Feature matrix

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize the data
    X_scaled = scaler.transform(X)

    return X_scaled