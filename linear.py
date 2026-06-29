import numpy as np
from typing import Tuple, List


def compute_cost_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
    """
    Computes the Mean Squared Error (MSE) cost with L2 regularization for multiple linear regression.
    
    Parameters:
    - X (np.ndarray): Feature matrix of shape (m, n)
    - y (np.ndarray): Target vector (real prices) of shape (m,)
    - w (np.ndarray): Weights vector of shape (n,)
    - b (float): Bias scalar
    - lambda_ (float): Regularization strength parameter
    
    Returns:
    - total_cost (float): The computed total cost (MSE + L2 penalty).
    """
    m = X.shape[0]  # Number of training examples (cars)
    
    # 1. PREDICTION: Matrix multiplication of features and weights, plus the bias
    f_wb = np.dot(X, w) + b
    
    # 2. BASE COST FUNCTION: Mean Squared Error (MSE)
    base_cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)
    
    # 3. REGULARIZATION: Adding L2 regularization penalty term (Ridge)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    
    return base_cost + reg_cost


def compute_gradient_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Computes the gradient (derivatives) for multiple linear regression with L2 regularization using vectorization.
    
    Parameters:
    - X (np.ndarray): Feature matrix of shape (m, n)
    - y (np.ndarray): Target vector (real prices) of shape (m,)
    - w (np.ndarray): Weights vector of shape (n,)
    - b (float): Bias scalar
    - lambda_ (float): Regularization strength parameter
    
    Returns:
    - dj_dw (np.ndarray): The gradient of the cost w.r.t. the weights (shape: n,)
    - dj_db (float): The gradient of the cost w.r.t. the bias
    """
    m, n = X.shape
    
    # 1. RESIDUALS: Calculate difference between predictions and real targets
    f_wb = np.dot(X, w) + b
    errors = f_wb - y
    
    # 2. BASE GRADIENTS: Vectorized partial derivatives for weights and bias
    dj_dw = (1 / m) * np.dot(X.T, errors)
    dj_db = (1 / m) * np.sum(errors)
    
    # 3. REGULARIZATION: Add L2 penalty derivative to the weights gradient
    dj_dw += (lambda_ / m) * w  # FIXED: Changed '+' to '*' for correct L2 math
    
    return dj_dw, dj_db


def gradient_descent_linear(
    X: np.ndarray, 
    y: np.ndarray, 
    w_in: np.ndarray, 
    b_in: float, 
    alpha: float, 
    num_iters: int, 
    lambda_: float = 0.1,  # FIXED: Moved default argument to the end to prevent SyntaxError
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Runs gradient descent to optimize w and b for multiple linear regression with L2 regularization.
    
    Parameters:
    - w_in (np.ndarray): Initial weights vector of shape (n,)
    - b_in (float): Initial bias scalar
    - alpha (float): Learning rate
    - num_iters (int): Maximum number of iterations to run
    - lambda_ (float): Regularization strength parameter
    - tolerance (float): Minimum cost reduction required to continue training
    
    Returns:
    - w (np.ndarray): Optimized weights vector of shape (n,)
    - b (float): Optimized bias scalar
    - J_history (list): History of cost values computed during training
    """
    w = np.copy(w_in)  # Avoid modifying the original weights array
    b = b_in
    J_history = []
    
    print("Starting Gradient Descent training for Price Prediction...")
    print("-" * 40)
    
    # Safely determine logging intervals to prevent ZeroDivisionError if num_iters < 5
    print_step = max(1, num_iters // 5)
    
    for i in range(num_iters):
        # 1. GRADIENT CALCULATION: Compute derivatives including lambda parameter
        dj_dw, dj_db = compute_gradient_linear(X, y, w, b, lambda_)  # FIXED: Added missing lambda_ argument
        
        # 2. PARAMETER UPDATE: Update parameters simultaneously using the learning rate
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # 3. COST TRACKING: Calculate and record the current regularized cost
        cost = compute_cost_linear(X, y, w, b, lambda_)
        J_history.append(cost)
        
        # 4. PROGRESS LOGGING: Print current cost metrics at scheduled intervals
        if i % print_step == 0 or i == num_iters - 1:
            print(f"Iteration {i:>5}: Cost = {cost:.4f}")
            
        # 5. EARLY STOPPING: Halt execution if cost reduction falls below threshold
        if i > 0 and abs(J_history[-2] - J_history[-1]) < tolerance:
            print("-" * 40)
            print(f"Convergence reached early at iteration {i:>5}. Cost = {cost:.4f}")
            break
            
    print("-" * 40)
    print("Training completed successfully!")
    
    return w, b, J_history