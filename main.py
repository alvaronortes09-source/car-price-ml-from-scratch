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
    # The CSV file is in the same folder, so we pass its exact name as an argument
    X, y_lin, y_log = preparing_cars_data('CarPrice_Assignment.csv')
    
    # Print execution summary to verify everything loaded correctly
    print("--- Data Preprocessing Completed ---")
    print(f"X matrix shape (samples, features): {X.shape}")
    print(f"Linear regression target shape: {y_lin.shape}")
    print(f"Logistic regression target shape: {y_log.shape}")