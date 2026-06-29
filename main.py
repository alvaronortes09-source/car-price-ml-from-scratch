import pandas as pd
import numpy as np
from linear import CustomLinearRegression
from logistic import CustomLogisticRegression
from utils import print_feature_importance, mae_score, r2_score       
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

    # ==============================================================================
    # 2. CUSTOM LOGISTIC REGRESSION (FUEL TYPE CLASSIFICATION)
    # ==============================================================================
    print("\n[INFO] Training Logistic Regression Model...")
    
    # Training the logistic regression model
    log_model = CustomLogisticRegression(alpha=alpha, iterations=iterations, lambda_reg=lambda_reg)
    log_model.fit(X_scaled, y_is_diesel)
    
    # Printng all the calculated parameters
    print(f" -> Final logistic bias (b): {log_model.b:.4f}")
    print_feature_importance(feature_names, log_model.w, pos_label="DIESEL", neg_label="GAS")
    
    # ==============================================================================
    # 3. CUSTOM LINEAR REGRESSION WITH RIDGE (L2) REGULARIZATION
    # ==============================================================================
    print("\n[TRAINING] Initializing Custom Linear Engine (Ridge L2)...")
    print("-" * 60)
    
    # Training the linear regression model
    lin_model = CustomLinearRegression(alpha=alpha, iterations=iterations, lambda_reg=lambda_reg)
    lin_model.fit(X_scaled, y_lin)
    
    #Prediction
    y_pred_custom = lin_model.predict(X_scaled)
    
    # Compute performance metrics
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
    
    #==============================================================================
    # 5. PRODUCTION BENCHMARK & FEATURE IMPORTANCE REPORT
    # ==============================================================================
    print("\n" + "═"*75)
    print("         MODEL AUDIT & FEATURE IMPORTANCE REPORT (CUSTOM VS SKLEARN)")
    print("═"*75)
    print(f" METRIC        | CUSTOM ENGINE       | SKLEARN BENCHMARK")
    print("-" * 75)
    # FIXED: Accedemos a lin_model.b en lugar de b_custom
    print(f" Mean Bias (b) | {lin_model.b:<19.4f} | {sk_model.intercept_:<19.4f}")
    print(f" MAE ($)       | ${mae_custom:<18.2f} | ${mae_sk:<18.2f}")
    print(f" R² Score      | {r2_custom:<19.4f} | {r2_sk:<19.4f}")
    print("=" * 75)
    
    print("\n[FEATURE IMPACT] Weights divergence check sorted by absolute magnitude:")
    print(f"{'FEATURE':<18} | {'CUSTOM WEIGHT':<15} | {'SKLEARN WEIGHT':<15} | {'PRICE IMPACT'}")
    print("-" * 75)
    
    sorted_features = sorted(zip(feature_names, lin_model.w, sk_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    
    for name, w_c, w_s in sorted_features:
        impact = "[+] Increases Price" if w_c > 0 else "[-] Decreases Price"
        print(f"{name:<18} | {w_c:<15.4f} | {w_s:<15.4f} | {impact}")
    print("=" * 75 + "\n")