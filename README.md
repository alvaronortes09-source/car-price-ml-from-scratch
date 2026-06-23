# Vectorized Car Price & Fuel Type Prediction from Scratch 🚗⚡

A 7-day Machine Learning challenge building core optimization algorithms from the ground up using only **NumPy** and **Pandas**. 

The goal of this project is to master the mathematical foundations of regression and classification models, vectorization, and L2 regularization without relying on Scikit-Learn or other high-level frameworks.

---

## 📊 Project Roadmap

- [x] **Day 1: Advanced Data Preprocessing & Target Isolation**
- [ ] **Day 2:** Vectorized Logistic Regression Cost & Gradient Functions
- [ ] **Day 3:** Gradient Descent Optimization for Fuel Type Classification
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

### Project Structure
```text
├── .gitignore
├── LICENSE
├── README.md
├── CarPrice_Assignment.csv
└── main.py
