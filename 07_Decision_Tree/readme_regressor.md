# Decision Tree Regressor (sklearn.tree.DecisionTreeRegressor) – Complete Guide

---

## 1. What is DecisionTreeRegressor?

`DecisionTreeRegressor` is a **supervised machine learning algorithm** used for **regression tasks**. It predicts continuous values by learning simple **decision rules** inferred from the features.

* Splits data recursively based on feature thresholds.
* Leaves represent predicted values (mean or median depending on criterion).
* Can handle **non-linear relationships**.
* Can overfit if not constrained with parameters like `max_depth` or `min_samples_leaf`.

---

## 2. Key Parameters

| Parameter                    | Meaning                                                         | Example / Notes                                                      |
| ---------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------- |
| **criterion**                | Function to measure split quality. Options:                     | `'squared_error'`, `'friedman_mse'`, `'absolute_error'`, `'poisson'` |
|                              | - `squared_error` → Mean squared error (L2 loss)                | Default, reduces variance                                            |
|                              | - `friedman_mse` → MSE with Friedman’s improvement score        | Can be better for boosting                                           |
|                              | - `absolute_error` → Mean absolute error (L1 loss, uses median) | Robust to outliers                                                   |
|                              | - `poisson` → Half mean Poisson deviance (for count data)       | Added in 0.24                                                        |
| **splitter**                 | Strategy to choose split at each node                           | `'best'` (default), `'random'`                                       |
| **max_depth**                | Maximum depth of the tree                                       | int, None → unlimited                                                |
| **min_samples_split**        | Minimum samples required to split an internal node              | int or float fraction, default=2                                     |
| **min_samples_leaf**         | Minimum samples required at a leaf node                         | int or float fraction, default=1                                     |
| **min_weight_fraction_leaf** | Minimum weighted fraction of samples at a leaf node             | float, default=0.0                                                   |
| **max_features**             | Number of features to consider when looking for the best split  | int, float, `'sqrt'`, `'log2'`, or None                              |
| **random_state**             | Seed for reproducibility                                        | int or None                                                          |
| **max_leaf_nodes**           | Maximum number of leaf nodes                                    | int, None → unlimited                                                |
| **min_impurity_decrease**    | Node will split if impurity decrease ≥ this value               | float, default=0.0                                                   |
| **ccp_alpha**                | Complexity parameter for Minimal Cost-Complexity Pruning        | float ≥ 0, default=0.0                                               |
| **monotonic_cst**            | Monotonicity constraint per feature                             | 1 (increasing), -1 (decreasing), 0 (none), or None                   |

---

## 3. Important Attributes After Fitting

| Attribute              | Meaning                                                                    |
| ---------------------- | -------------------------------------------------------------------------- |
| `feature_importances_` | Importance of each feature in prediction                                   |
| `max_features_`        | Inferred value of `max_features`                                           |
| `n_features_in_`       | Number of features seen during fit                                         |
| `feature_names_in_`    | Feature names seen during fit (if X has string names)                      |
| `n_outputs_`           | Number of outputs                                                          |
| `tree_`                | Underlying Tree object; contains the structure of the fitted decision tree |

---

## 4. Key Methods

| Method                                      | Description                                                            |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| `fit(X, y)`                                 | Train the decision tree on input `X` and target `y`.                   |
| `predict(X)`                                | Predict regression values for samples in `X`.                          |
| `score(X, y)`                               | Return R² (coefficient of determination) on test data.                 |
| `get_depth()`                               | Return maximum depth of the tree.                                      |
| `get_n_leaves()`                            | Return number of leaves in the tree.                                   |
| `apply(X)`                                  | Return the index of the leaf each sample ends up in.                   |
| `decision_path(X)`                          | Return sparse indicator matrix showing which nodes samples go through. |
| `cost_complexity_pruning_path(X, y)`        | Compute effective alphas for minimal cost-complexity pruning.          |
| `get_params()` / `set_params()`             | Get or set estimator parameters (useful in pipelines/grid search).     |
| `set_fit_request()` / `set_score_request()` | Configure metadata routing for fit or score methods (advanced usage).  |

---

## 5. Notes

* Fully grown trees can **overfit** on small datasets; control size with `max_depth`, `min_samples_leaf`, or `ccp_alpha`.
* Supports **non-linear regression**, handles both numerical and categorical features (after encoding).
* Can handle **multi-output regression**, except when using monotonic constraints.
* Works well in ensemble methods like **Random Forest** or **Gradient Boosted Trees**.

---

## 6. Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Initialize regressor
regressor = DecisionTreeRegressor(random_state=0)

# Evaluate with cross-validation
scores = cross_val_score(regressor, X, y, cv=10)
print(scores)

# Fit model and make predictions
regressor.fit(X, y)
preds = regressor.predict(X[:5])
print(preds)

# Get leaf indices
leaves = regressor.apply(X[:5])
print(leaves)
```

