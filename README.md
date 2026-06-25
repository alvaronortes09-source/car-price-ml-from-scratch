# Vectorized Car Price & Fuel Type Prediction from Scratch 🚗⚡

A 7-day Machine Learning challenge building core optimization algorithms from the ground up using only **NumPy** and **Pandas**. 

The goal of this project is to master the mathematical foundations of regression and classification models, vectorization, and L2 regularization without relying on Scikit-Learn or other high-level frameworks.

---

## 📊 Project Roadmap

- [x] **Day 1: Advanced Data Preprocessing & Target Isolation**
- [x] **Day 2: Vectorized Logistic Regression Cost & Gradient Engines**
- [x] **Day 3: Gradient Descent Optimization & Model Interpretability**
- [ ] **Day 4:** Linear Regression Implementation for Price Prediction
- [ ] **Day 5:** Feature Engineering & L2 Regularization (Ridge Penalty)
- [ ] **Day 6:** Model Evaluation & Scikit-Learn Benchmarking
- [ ] **Day 7:** Production-Ready Refactoring into Object-Oriented Programming (OOP)

---

## 🛠️ Day 1 Summary: Preprocessing Pipeline

Today I built the data ingestion and normalization pipeline in `main.py` using a strict vectorization approach.

### Key Implementations:
1. **Data Ingestion:** Loaded the CarPrice dataset using Pandas to parse tabular data.
2. **Categorical Encoding:** Transformed the string column `fueltype` into a binary numeric target (`is_diesel`: `1` for diesel, `0` for gas) using vectorized boolean casting.
3. **Feature Scaling (Z-score Normalization):** Implemented feature scaling entirely in NumPy to ensure numerical stability during gradient descent:
   $$X_{scaled} = \frac{X - \mu}{\sigma}$$
4. **Target Isolation:** Separated the dataset into feature matrix $X$ and isolated targets for both continuous prediction (`price`) and binary classification (`is_diesel`).

---

## 🧠 Day 2 Summary: Core Logistic Regression Engine

On Day 2, the core mathematical and optimization foundation of the Logistic Regression model was fully implemented from scratch using NumPy. The focus was shifted from data understanding to building an efficient, stable, and production-ready training architecture.

### Advanced Engineering Decisions & Rationale

Building a machine learning model from scratch introduces real-world software bottlenecks. Below are the advanced implementations introduced to elevate the codebase to production standards:

* **Vectorized Cost Computation (`np.dot` & `np.sum`)**
  * *The Challenge:* Standard implementation tutorials use nested `for` loops to iterate through features and samples. In Python, native loops introduce massive computational overhead, causing the script to lag exponentially as the dataset scales.
  * *The Solution:* The cost function was completely vectorized. By shifting the math to matrix operations, the execution is delegated to NumPy's underlying C-optimized libraries, resulting in near-instantaneous execution.
* **Numerical Stability Shield (`np.clip`)**
  * *The Challenge:* During early training stages or when using aggressive learning rates, the sigmoid activation function can output extreme probabilities close to exactly `0.0` or `1.0`. Mathematically, evaluating $\log(0)$ yields negative infinity ($-\infty$), which injects `NaN` (Not a Number) values into the weights and permanently corrupts the training process.
  * *The Solution:* Implemented numerical clipping to bound probability outputs strictly within $[10^{-15}, 1 - 10^{-15}]$. This guarantees mathematical safety without compromising model accuracy.
* **Algorithmic Efficiency via Early Stopping (`tolerance`)**
  * *The Challenge:* Running a fixed number of iterations (e.g., 10,000) wastes valuable server runtime and CPU cycles if the model reaches its optimal parameters early. Once the cost curve flattens, further gradient calculations are redundant.
  * *The Solution:* Added a convergence monitoring mechanism. By measuring the absolute difference between the current cost and the previous iteration's cost against a strict threshold (`1e-6`), the loop breaks automatically upon convergence, significantly reducing compute costs.
* **Enterprise-Grade Architecture via Static Type Hinting**
  * *The Challenge:* Python's dynamically-typed nature often causes silent runtime type errors in data pipelines (e.g., passing a list instead of a NumPy matrix), making large codebases hard to maintain and debug.
  * *The Solution:* Enforced strict type annotations (`np.ndarray`, `Tuple`, `List`) across all mathematical components. This bridges the gap between basic script writing and production software architecture, ensuring robust code integration and IDE safety.

### Implemented Functions

The core module now contains the following functional components:

1. `sigmoid(z)`: Maps linear predictions into probabilities ranging between 0 and 1.
2. `compute_cost_logistic(X, y, w, b)`: Computes the Binary Cross-Entropy loss using a fully vectorized approach.
3. `compute_gradient_logistic(X, y, w, b)`: Calculates the partial derivatives (gradients) with respect to weights (`w`) and bias (`b`) to determine the steepest descent direction.
4. `gradient_descent_logistic(X, y, w_in, b_in, alpha, num_iters, tolerance)`: The main execution loop that updates model weights iteratively, tracks the cost history, and supports early termination.

### Mathematical Framework

The optimization process evaluates the model's error utilizing the **Binary Cross-Entropy Loss** function:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{w,b}(X^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(X^{(i)})) \right]$$

Where the classification prediction is defined by the sigmoid activation function:

$$f_{w,b}(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

---

## 🚀 Day 3 Summary: Model Training & Feature Interpretability Pipeline

On Day 3, the preprocessing pipeline from Day 1 and the mathematical engine from Day 2 were integrated into a unified execution flow. The model was trained successfully, and a decoupled presentation layer was designed to extract production-grade model insights.

### Advanced Engineering Decisions & Rationale

* **Decoupled Presentation Layer via Reusable Utilities (`utils.py`)**
  * *The Challenge:* Mixing pure data science workflows (training loops) with UI/formatting code inside `main.py` violates the Separation of Concerns (SoC) principle, turning the file into a monolithic script that is difficult to maintain and scale.
  * *The Solution:* Extracted the formatting, padding, and alignment logic into a generic function (`print_feature_importance`) inside a new dedicated `utils.py` module. This allows cross-project reuse and clean, isolated codebases.
* **Robust Production Tabular Formatting**
  * *The Challenge:* Features vary in character length (e.g., `peakrpm` vs `compressionratio`), which causes standard stdout print blocks to shift irregularly, making logs messy and unprofessional for engineering teams or recruiters.
  * *The Solution:* Designed strict string formatting blocks (`{name:<22}` and `{weight:>15.4f}`) ensuring uniform column boundaries across different operating systems and terminal encodings, intentionally avoiding unstable non-UTF-8 visual elements (emojis).
* **Mathematical Sorting by Absolute Magnitude**
  * *The Challenge:* Features are naturally ordered based on the original dataset columns. Displaying them raw requires manual sorting to understand which factor dominates the network's behavior.
  * *The Solution:* Engineered a custom sorting key using Python lambda structures (`key=lambda x: abs(x[1])`) to dynamically rank weights by absolute mathematical impact, highlighting top statistical triggers at the apex of the report.

### Training Performance & Convergence Dynamics

The optimization process achieved high efficiency, validating the math engine against the engine metrics:

```text
Starting Gradient Descent training...
----------------------------------------
Iteration     0: Cost = 0.6661
Iteration  1000: Cost = 0.0107
Iteration  2000: Cost = 0.0054
Iteration  3000: Cost = 0.0036
Convergence reached early at iteration  3269. Cost = 0.0033
----------------------------------------
Training completed successfully!
Final optimized bias (b): -4.7741
