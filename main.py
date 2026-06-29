import pandas as pd
import numpy as np
from logistic import gradient_descent_logistic
from linear import gradient_descent_linear
from utils import print_feature_importance
from utils import mae_score, r2_score       
from sklearn.linear_model import Ridge as SklearnRidge

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
    lambda_reg = 10.0

    # --- Logistic Regression ---
    print("\n[INFO] Training Logistic Regression Model...")
    w_init_log = np.zeros(num_features,)
    b_init_log = 0.0
    
    w_logistic, b_logistic, J_hist_log = gradient_descent_logistic(
        X_scaled, y_is_diesel, w_init_log, b_init_log, alpha, iterations
    )
    
    print(f" -> Final logistic bias (b): {b_logistic:.4f}")
    print_feature_importance(feature_names, w_logistic, pos_label="DIESEL", neg_label="GAS")
    
# ==============================================================================
    # 3. CUSTOM LINEAR REGRESSION WITH RIDGE (L2) REGULARIZATION
    # ==============================================================================
    print("\n[TRAINING] Initializing Custom Linear Engine (Ridge L2)...")
    print("-" * 60)
    
    # Initialize weights and bias to zero before starting the optimization loop
    w_init_lin = np.zeros(num_features,)
    b_init_lin = 0.0
    
    # Execute gradient descent to find optimal parameters (w, b)
    w_custom, b_custom, _ = gradient_descent_linear(
        X_scaled, y_lin, w_init_lin, b_init_lin, alpha, iterations, lambda_=lambda_reg
    )
    
    # Generate predictions using our optimized custom weights
    y_pred_custom = np.dot(X_scaled, w_custom) + b_custom
    
    # Compute performance metrics to evaluate the model against the ground truth
    mae_custom = mae_score(y_lin, y_pred_custom)
    r2_custom = r2_score(y_lin, y_pred_custom)

    # ==============================================================================
    # 4. INDUSTRY BENCHMARK: SCIKIT-LEARN VALIDATION
    # ==============================================================================
    print("\n[AUDIT] Training Scikit-Learn Ridge model for benchmark validation...")
    
    # Calibrate the regularization strength (alpha) for Scikit-Learn.
    # Sklearn does not divide the penalty by 2m in its internal cost function by default.
    # We apply this scaling factor to ensure a mathematically identical comparison.
    sklearn_alpha = lambda_reg / (2 * len(y_lin)) 
    
    # Instantiate and fit the production-grade model using Cholesky decomposition
    sk_model = SklearnRidge(alpha=sklearn_alpha, solver='cholesky')
    sk_model.fit(X_scaled, y_lin)
    
    # Generate benchmark predictions and extract baseline metrics
    y_pred_sk = sk_model.predict(X_scaled)
    mae_sk = mae_score(y_lin, y_pred_sk)
    r2_sk = r2_score(y_lin, y_pred_sk)
    
# ==============================================================================
    # 5. PRODUCTION BENCHMARK & FEATURE IMPORTANCE REPORT
    # ==============================================================================
    # Output a side-by-side technical audit comparing our custom mathematical engine 
    # against the industry standard (Scikit-Learn).
    
    print("\n" + "═"*75)
    print("       MODEL AUDIT & FEATURE IMPORTANCE REPORT (CUSTOM VS SKLEARN)")
    print("═"*75)
    print(f" METRIC        | CUSTOM ENGINE       | SKLEARN BENCHMARK")
    print("-" * 75)
    print(f" Mean Bias (b) | {b_custom:<19.4f} | {sk_model.intercept_:<19.4f}")
    print(f" MAE ($)       | ${mae_custom:<18.2f} | ${mae_sk:<18.2f}")
    print(f" R² Score      | {r2_custom:<19.4f} | {r2_sk:<19.4f}")
    print("=" * 75)
    
    print("\n[FEATURE IMPACT] Weights divergence check sorted by absolute magnitude:")
    print(f"{'FEATURE':<18} | {'CUSTOM WEIGHT':<15} | {'SKLEARN WEIGHT':<15} | {'PRICE IMPACT'}")
    print("-" * 75)
    
    # Pack feature names with their respective custom and sklearn weights.
    # Sort the tuples in descending order based on the absolute value of our custom weight (x[1]).
    # This prioritizes variables that have the highest magnitude of impact on the target (price).
    sorted_features = sorted(zip(feature_names, w_custom, sk_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    
    # Iterate through the sorted features to map raw mathematical weights to business logic.
    for name, w_c, w_s in sorted_features:
        # Determine price directionality: positive weights inflate price, negative weights deflate it.
        impact = "[+] Increases Price" if w_c > 0 else "[-] Decreases Price"
        print(f"{name:<18} | {w_c:<15.4f} | {w_s:<15.4f} | {impact}")
    print("=" * 75 + "\n")