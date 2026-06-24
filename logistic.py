from typing import Tuple, List
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z.
    """
    # Apply the element-wise sigmoid activation formula
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Compute the cost for logistic regression using a vectorized implementation.
    Includes clipping for numerical stability.
    """
    m = X.shape[0]
    
    # Calculate linear combinations and predictions for all samples simultaneously
    Z = np.dot(X, w) + b
    A = sigmoid(Z)
    
    # Prevent log(0) undefined errors by bounding probabilities slightly away from 0 and 1
    A = np.clip(A, 1e-15, 1 - 1e-15)
    
    # Compute the average binary cross-entropy loss across the entire dataset
    cost = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    return float(cost)

def compute_gradient_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Compute the gradient for logistic regression using explicit loops.
    """
    m, n = X.shape
    # Initialize gradient accumulations for weights and bias
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    
    # Loop through each individual training sample
    for i in range(m):
        # Calculate prediction and prediction error for the current sample
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        
        # Accumulate derivatives for each individual feature coefficient
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
            
        # Accumulate the bias derivative component
        dj_db = dj_db + err_i
        
    # Scale the accumulated gradients by the total number of training examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent_logistic(
    X: np.ndarray, 
    y: np.ndarray, 
    w_in: np.ndarray, 
    b_in: float, 
    alpha: float, 
    num_iters: int,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Perform batch gradient descent to optimize parameters w and b.
    Includes an early stopping mechanism based on cost convergence tolerance.
    """
    # Track the optimization progress and prevent side effects on input parameters
    J_history = []
    w = np.copy(w_in)
    b = b_in
    
    for i in range(num_iters):
        # Evaluate partial derivatives at the current position
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        
        # Update parameters simultaneously by taking a step against the gradient direction
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
        # Record current cost to monitor model performance over iterations
        cost = compute_cost_logistic(X, y, w, b)
        J_history.append(cost)
        
        # Stop execution if the cost reduction falls below the specified threshold
        if i > 0 and (J_history[-2] - cost) < tolerance:
            print(f"Convergence reached early at iteration {i:5d}. Cost = {cost:.4f}")
            break
        
        # Print status updates at 20% milestones of the optimization process
        if i % (num_iters // 5) == 0:
            print(f"Iteration {i:5d}: Cost = {cost:.4f}")
        
    return w, b, J_history