import numpy as np
from typing import Tuple, List

class CustomLinearRegression:
    def __init__(self, alpha: float = 0.1, iterations: int = 1000, lambda_reg: float = 0.0):
        #Hyperparameters
        self.alpha = alpha
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        
        #Model parameters
        self.w = None
        self.b = None
        self.cost_history = []
        
    def compute_cost_linear(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
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


    def compute_gradient_linear(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> Tuple[np.ndarray, float]:
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
        m = X.shape[0]
    
        # 1. RESIDUALS: Calculate difference between predictions and real targets
        f_wb = np.dot(X, w) + b
        errors = f_wb - y
    
        # 2. BASE GRADIENTS: Vectorized partial derivatives for weights and bias
        dj_dw = (1 / m) * np.dot(X.T, errors)
        dj_db = (1 / m) * np.sum(errors)
    
        # 3. REGULARIZATION: Add L2 penalty derivative to the weights gradient
        dj_dw += (lambda_ / m) * w 
    
        return dj_dw, dj_db

    def fit(self, X: np.ndarray, y: np.ndarray, tolerance: float = 1e-6):
        """Fits the model using vectorized gradient descent."""
        num_features = X.shape[1]
        self.w = np.zeros(num_features,)
        self.b = 0.0
        self.cost_history = []
        
        #Gradient descent for linear regression
        
        # Safely determine logging intervals to prevent ZeroDivisionError if num_iters < 5
        print_step = max(1, self.iterations // 5)
        
        for i in range(self.iterations):
            # 1. GRADIENT CALCULATION: Compute derivatives including lambda parameter
            dj_dw, dj_db = self.compute_gradient_linear(X, y, self.w, self.b)
            
            # 2. PARAMETER UPDATE: Update parameters simultaneously using the learning rate
            self.w = self.w - self.alpha * dj_dw
            self.b = self.b - self.alpha * dj_db
        
            # 3. COST TRACKING: Calculate and record the current regularized cost
            cost = self.compute_cost_linear(X, y, self.w, self.b)
            self.cost_history.append(cost)
        
            # 4. PROGRESS LOGGING: Print current cost metrics at scheduled intervals
            if i % print_step == 0 or i == self.iterations - 1:
                print(f"Iteration {i:>5}: Cost = {cost:.4f}")
            
            # 5. EARLY STOPPING: Halt execution if cost reduction falls below threshold
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < tolerance:
                print("-" * 40)
                print(f"Convergence reached early at iteration {i:>5}. Cost = {cost:.4f}")
                break
            
        print("-" * 40)
        print("Training completed successfully!")
    
        return self
        
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates continuous price predictions."""
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before running predictions.")
        return np.dot(X, self.w) + self.b