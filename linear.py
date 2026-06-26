import numpy as np

def compute_cost_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes the Mean Squared Error (MSE) cost for multiple linear regression.
    
    Parameters:
    - X (np.ndarray): Feature matrix of shape (m, n) -> e.g., (205, 5)
    - y (np.ndarray): Target vector (real prices) of shape (m,) -> e.g., (205,)
    - w (np.ndarray): Weights vector of shape (n,) -> e.g., (5,)
    - b (float): Bias scalar
    
    Returns:
    - total_cost (float): The computed MSE loss.
    """
    m = X.shape[0]  # Number of training examples (cars)
    
    # 1. PREDICTION: Matrix multiplication of features and weights, plus the bias
    f_wb = np.dot(X, w) + b
    
    # 2. ERROR: Element-wise subtraction between model predictions and real prices
    errors = f_wb - y
    
    # 3. COST: Square the errors (**2), sum them up, and average them out
    total_cost = (1 / (2 * m)) * np.sum(errors ** 2)
    
    return total_cost


def compute_gradient_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Computes the gradient (derivatives) for multiple linear regression using vectorization.
    
    Parameters:
    - X (np.ndarray): Feature matrix of shape (m, n) -> e.g., (205, 5)
    - y (np.ndarray): Target vector (real prices) of shape (m,) -> e.g., (205,)
    - w (np.ndarray): Weights vector of shape (n,) -> e.g., (5,)
    - b (float): Bias scalar
    
    Returns:
    - dj_dw (np.ndarray): The gradient of the cost w.r.t. the weights (shape: n,)
    - dj_db (float): The gradient of the cost w.r.t. the bias
    """
    m, n = X.shape
    
    f_wb = np.dot(X, w) + b
    errors = f_wb - y
    
    dj_dw = (1 / m) * np.dot(X.T, errors)
    
    dj_db = (1 / m) * np.sum(errors)
    
    return dj_dw, dj_db

import numpy as np
from typing import Tuple, List

def gradient_descent_linear(
    X: np.ndarray, 
    y: np.ndarray, 
    w_in: np.ndarray, 
    b_in: float, 
    alpha: float, 
    num_iters: int, 
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Runs gradient descent to optimize w and b for multiple linear regression.
    
    Parameters:
    - w_in (np.ndarray): Initial weights vector of shape (n,)
    - b_in (float): Initial bias scalar
    - alpha (float): Learning rate
    - num_iters (int): Maximum number of iterations to run
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
    
    for i in range(num_iters):
        # 1. Calculate the gradient (slopes) using our vectorized function
        dj_dw, dj_db = compute_gradient_linear(X, y, w, b)
        
        # 2. Update parameters simultaneously using the learning rate (alpha)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # 3. Calculate and record the current cost
        cost = compute_cost_linear(X, y, w, b)
        J_history.append(cost)
        
        # 4. Print progress every 10% of iterations
        if i % (num_iters // 5) == 0 or i == num_iters - 1:
            print(f"Iteration {i:>5}: Cost = {cost:.4f}")
            
        # 5. Early Stopping Check (Algorithmic Efficiency)
        if i > 0 and abs(J_history[-2] - J_history[-1]) < tolerance:
            print("-" * 40)
            print(f"Convergence reached early at iteration {i:>5}. Cost = {cost:.4f}")
            break
            
    print("-" * 40)
    print("Training completed successfully!")
    
    return w, b, J_history