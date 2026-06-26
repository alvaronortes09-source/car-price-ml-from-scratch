import pandas as pd
import numpy as np
from logistic import gradient_descent_logistic
from linear import gradient_descent_linear
from utils import print_feature_importance

def preparing_cars_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Encode target variable: 1 (diesel), 0 (gas)
    df['is_diesel'] = (df['fueltype'] == 'diesel').astype(int)
    
    # Select continuous features
    columns_X = ['horsepower', 'enginesize', 'curbweight', 'compressionratio', 'peakrpm']
    X = df[columns_X].values
    
    # Z-score normalization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # Extract targets
    y_linear = df['price'].values
    y_logistic = df['is_diesel'].values
    
    return X_scaled, y_linear, y_logistic


if __name__ == "__main__":
    # --- Data Pipeline ---
    X_scaled, y_lin, y_is_diesel = preparing_cars_data('CarPrice_Assignment.csv')
    feature_names = ['horsepower', 'enginesize', 'curbweight', 'compressionratio', 'peakrpm']
    num_features = X_scaled.shape[1]
    
    print("\n" + "="*60)
    print("          AUTOMOTIVE ML PIPELINE - INITIALIZATION")
    print("="*60)
    print(f" -> X Matrix (Samples, Features) : {X_scaled.shape}")
    print(f" -> Linear Target (Prices)       : {y_lin.shape}")
    print(f" -> Logistic Target (Fuel Type)  : {y_is_diesel.shape}")
    print("-" * 60)
    
    # Hyperparameters
    alpha = 0.1
    iterations = 5000

    # --- Logistic Regression ---
    print("\n[INFO] Training Logistic Regression Model...")
    w_init_log = np.zeros(num_features,)
    b_init_log = 0.0
    
    w_logistic, b_logistic, J_hist_log = gradient_descent_logistic(
        X_scaled, y_is_diesel, w_init_log, b_init_log, alpha, iterations
    )
    
    print(f" -> Final logistic bias (b): {b_logistic:.4f}")
    print_feature_importance(feature_names, w_logistic, pos_label="DIESEL", neg_label="GAS")


    # --- Linear Regression ---
    print("\n[INFO] Training Linear Regression Model...")
    w_init_lin = np.zeros(num_features,)
    b_init_lin = 0.0
    
    w_linear, b_linear, J_hist_lin = gradient_descent_linear(
        X_scaled, y_lin, w_init_lin, b_init_lin, alpha, iterations
    )
    
    print(f" -> Final linear bias (b): {b_linear:.4f}")
    
    # Display feature weights sorted by absolute magnitude
    print("\n============================================================")
    print(f"{'FEATURE NAME':<22} | {'OPTIMIZED WEIGHT':<16} | {'IMPACT ON PRICE'}")
    print("============================================================")
    
    sorted_features = sorted(zip(feature_names, w_linear), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, weight in sorted_features:
        impact = "[+] Increases Price" if weight > 0 else "[-] Decreases Price"
        print(f"{feature:<22} | {weight:>16.4f} | {impact}")
    print("============================================================\n")