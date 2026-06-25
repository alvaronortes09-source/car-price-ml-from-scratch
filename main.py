import pandas as pd
import numpy as np

# Prepare the dataset for Machine Learning processing
def preparing_cars_data(file_path):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Create a binary target variable for Logistic Regression (1 for diesel, 0 for gas)
    df['is_diesel'] = (df['fueltype'] == 'diesel').astype(int)
    
    # Select specific continuous features to train our models
    columns_X = ['horsepower', 'enginesize', 'curbweight', 'compressionratio', 'peakrpm']
    X = df[columns_X].values
    
    # Apply Z-score Normalization (Feature Scaling)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaling = (X - X_mean) / X_std
    
    # Extract the target variables into separate NumPy arrays
    y_lineal = df['price'].values          # Continuous target for Linear Regression
    y_logistic = df['is_diesel'].values    # Binary target for Logistic Regression
    
    # Crucial step: return the processed data to the main program
    return X_scaling, y_lineal, y_logistic

if __name__ == "__main__":
    # The CSV file is in the same folder, so we give its exact name as an argument
    X_scaled, y_lin, y_is_diesel = preparing_cars_data('CarPrice_Assignment.csv')
    feature_names = ['horsepower', 'enginesize', 'curbweight', 'compressionratio', 'peakrpm']
    
    # Print execution summary to verify everything loaded correctly
    print("--- Data Preprocessing Completed ---")
    print(f"X matrix shape (samples, features): {X_scaled.shape}")
    print(f"Linear regression target shape: {y_lin.shape}")
    print(f"Logistic regression target shape: {y_is_diesel.shape}")
    

# ==========================================
# TRAINING THE LOGISTIC REGRESSION MODEL
# ==========================================

from logistic import gradient_descent_logistic

# 1. Initialize parameters as floats
num_features = X_scaled.shape[1]
w_init = np.zeros(num_features,)
b_init = 0.0

# 2. Set of hyperparameters
alpha = 0.1
iterations = 5000

print("Starting Gradient Descent training...")
print("-" * 40)

w_final, b_final, J_history = gradient_descent_logistic(
    X_scaled,
    y_is_diesel,
    w_init,
    b_init,
    alpha,
    iterations
)

print("-" * 40)
print("Training completed successfully!")
print(f"Final optimized bias (b): {b_final:.4f}")

# =====================================================================
# Feature Importance & Model Interpretability
# =====================================================================
from utils import print_feature_importance

print_feature_importance(feature_names, w_final, pos_label="DIESEL", neg_label="GAS")