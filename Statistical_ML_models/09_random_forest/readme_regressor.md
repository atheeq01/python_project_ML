# RandomForestRegressor — scikit-learn

`sklearn.ensemble.RandomForestRegressor` is a **meta-estimator** that fits multiple **decision tree regressors** on different sub-samples of the dataset and averages their predictions to improve accuracy and reduce over-fitting.

---

## 1. **Overview**

- A **random forest** is an ensemble of decision trees for regression.  
- Each tree is trained on a **bootstrapped sample** of the dataset (if `bootstrap=True`).  
- Predictions are made by **averaging** outputs of all trees.  
- Trees use the **best split strategy**, equivalent to `splitter="best"` in `DecisionTreeRegressor`.  
- Supports **missing values (NaN)**. During training, the tree learns whether samples with missing values should go left or right.

---

## 2. **Key Parameters**

| Parameter | Default | Typical Range / Values | Description |
|-----------|--------|----------------------|-------------|
| `n_estimators` | 100 | 10–1000+ | Number of trees in the forest. More trees → better predictions, slower training. |
| `criterion` | "squared_error" | "squared_error", "absolute_error", "friedman_mse", "poisson" | Function to measure split quality. |
| `max_depth` | None | 2–30 | Maximum depth of each tree. None = fully grown until leaves are pure or `min_samples_split` reached. |
| `min_samples_split` | 2 | 2–20 | Minimum samples required to split an internal node (int or fraction). |
| `min_samples_leaf` | 1 | 1–10 | Minimum samples required at a leaf node. |
| `min_weight_fraction_leaf` | 0.0 | 0.0–0.5 | Minimum weighted fraction of the sum of sample weights required at a leaf. |
| `max_features` | 1.0 | int, float, "sqrt", "log2", None | Number of features to consider when looking for the best split. |
| `max_leaf_nodes` | None | int or None | Maximum number of leaf nodes. None = unlimited. |
| `min_impurity_decrease` | 0.0 | 0.0–0.1 | Node is split only if impurity decreases at least by this value. |
| `bootstrap` | True | True / False | Whether bootstrap samples are used for training. |
| `oob_score` | False | True / False | Whether to use out-of-bag samples to estimate generalization score. |
| `n_jobs` | None | int or -1 | Number of jobs to run in parallel. -1 uses all processors. |
| `random_state` | None | int | Controls randomness in bootstrapping and feature selection. |
| `verbose` | 0 | 0–3 | Controls verbosity of training. |
| `warm_start` | False | True / False | Reuse previous solution to add more trees. |
| `ccp_alpha` | 0.0 | 0.0–1.0 | Complexity parameter for Minimal Cost-Complexity Pruning. |
| `max_samples` | None | int or float | Number of samples to draw from X for each tree when `bootstrap=True`. |
| `monotonic_cst` | None | array-like | Enforces monotonic constraints on features (1=increasing, -1=decreasing). |

---

## 3. **Important Attributes**

| Attribute | Description |
|-----------|-------------|
| `estimators_` | List of fitted `DecisionTreeRegressor` objects. |
| `feature_importances_` | Impurity-based feature importances. |
| `n_features_in_` | Number of features seen during fit. |
| `feature_names_in_` | Names of features seen during fit (if X has feature names). |
| `n_outputs_` | Number of outputs. |
| `oob_score_` | Out-of-bag R² score (if `oob_score=True`). |
| `oob_prediction_` | Predictions on training data using OOB samples. |
| `estimators_samples_` | Subset of samples used for each base estimator. |

---

## 4. **Key Methods**

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the random forest regressor on data `X` with targets `y`. |
| `predict(X)` | Predict continuous target values for samples in `X` (mean of all trees). |
| `score(X, y)` | Compute R² coefficient of determination. |
| `apply(X)` | Return leaf index for each sample in each tree. |
| `decision_path(X)` | Return a sparse matrix indicating the nodes each sample goes through. |
| `set_params(**params)` | Set estimator parameters. |
| `get_params(deep=True)` | Get estimator parameters. |

---

## 5. **How Random Forest Regression Works**

1. Draw `n_estimators` **bootstrap samples** from the dataset.  
2. Train a **decision tree regressor** on each sample considering `max_features` features at each split.  
3. Each tree predicts independently; the forest outputs the **average prediction**.  
4. Optionally, **OOB samples** estimate accuracy without a separate validation set.

---

## 6. **Real-World Examples**

| Scenario | Input Features | Target | Description |
|----------|----------------|--------|-------------|
| House Price Prediction | Area, bedrooms, location, age | Price | Predict house prices. |
| Stock Forecasting | Previous prices, volumes, indicators | Future price | Predict next-day stock prices. |
| Energy Consumption | Temperature, humidity, occupancy | Energy usage | Predict electricity demand. |
| Sales Prediction | Advertising spend, seasonality, product type | Sales | Predict monthly sales of a product. |

---

## 7. **Example Code**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=4,
                       n_informative=2, noise=0.1, random_state=0)

# Create regressor
regr = RandomForestRegressor(max_depth=3, random_state=0)

# Train the model
regr.fit(X, y)

# Predict new sample
print(regr.predict([[0, 0, 0, 0]]))  # e.g., [-8.33]
