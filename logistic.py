from typing import Tuple, List
import numpy as np

class CustomLogisticRegression:
    def __init__(self, alpha: float = 0.1, iterations: int = 1000, lambda_reg: float = 0.0):
        #Hyperparameters
        self.alpha = alpha
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        
        #Model parameters
        self.w = None
        self.b = None
        self.cost_history = []
        

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of z.
        """
        # Apply the element-wise sigmoid activation formula
        g = 1 / (1 + np.exp(-z))
        return g

    def compute_cost_logistic(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        """
        Compute the cost for logistic regression using a vectorized implementation.
        Includes clipping for numerical stability and L2 Regularization.
        """
        m = X.shape[0]
        Z = np.dot(X, w) + b
        A = self.sigmoid(Z)
        
        # Prevent log(0) undefined errors
        A = np.clip(A, 1e-15, 1 - 1e-15)
    
        # Cross-entropy loss + L2 regularization penalty
        cross_entropy = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        l2_penalty = (self.lambda_reg / (2 * m)) * np.sum(w ** 2)
        
        return float(cross_entropy + l2_penalty)

    def compute_gradient_logistic(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
        """
        Compute the gradient for logistic regression using FULL vectorization.
        No more slow explicit loops.
        """
        m = X.shape[0]
        
        # 1. Predictions vector (probabilities)
        Z = np.dot(X, w) + b
        A = self.sigmoid(Z)
        
        # 2. Error vector
        error = A - y
        
        # 3. Vectorized gradients with L2 Regularization
        dj_dw = (1 / m) * np.dot(X.T, error) + (self.lambda_reg / m) * w
        dj_db = (1 / m) * np.sum(error)
    
        return dj_dw, dj_db
    
    def fit(self, X: np.ndarray, y: np.ndarray, tolerance: float = 1e-6):
        """Fits the logistic model using vectorized gradient descent."""
        num_features = X.shape[1]
        
        self.w = np.zeros(num_features,)
        self.b = 0.0
        self.cost_history = []
        
        print("Starting Gradient Descent training for Fuel Type Classification...")
        print("-" * 40)
        
        # Safely determine logging intervals to prevent ZeroDivisionError if num_iters < 5
        print_step = max(1, self.iterations // 5)
        
        for i in range(self.iterations):
            # Evaluate partial derivatives at the current position
            dj_dw, dj_db = self.compute_gradient_logistic(X, y, self.w, self.b)
        
            # Update parameters simultaneously by taking a step against the gradient direction
            self.w = self.w - self.alpha * dj_dw
            self.b = self.b - self.alpha * dj_db
    
            # Record current cost to monitor model performance over iterations
            cost = self.compute_cost_logistic(X, y, self.w, self.b)
            self.cost_history.append(cost)
        
            # Stop execution if the cost reduction falls below the specified threshold
            if i % print_step == 0 or i == self.iterations - 1:
                print(f"Iteration {i:>5}: Cost = {cost:.4f}")
        
            # Early stopping check using the local tolerance variable
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < tolerance:
                print("-" * 40)
                print(f"Convergence reached early at iteration {i:>5}. Cost = {cost:.4f}")
                break
                
        print("-" * 40)
        print("Training completed successfully!")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the raw probability (0 to 1) of the positive class."""
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before running predictions.")
            
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns discrete binary classifications (1 or 0)."""
        probabilities = self.predict_proba(X)
        # Vectorization: if X is >= threshold is True (1), else is False (0)
        return (probabilities >= threshold).astype(int)