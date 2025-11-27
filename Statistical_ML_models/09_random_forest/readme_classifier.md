# RandomForestClassifier — scikit-learn

`sklearn.ensemble.RandomForestClassifier` is a **meta-estimator** that fits multiple **decision tree classifiers** on different sub-samples of the dataset and combines their predictions to improve accuracy and control over-fitting.

---

## 1. **Overview**

- A **random forest** is an ensemble of decision trees.  
- Each tree is trained on a **bootstrapped sample** of the dataset (if `bootstrap=True`).  
- The final prediction is made by **majority voting** across all trees.  
- Trees use the **best split strategy**, equivalent to `splitter="best"` in `DecisionTreeClassifier`.  
- Handles **missing values (NaN)** by learning how to send them left or right at each split.

---

## 2. **Key Parameters**

| Parameter | Default | Typical Range / Values | Description |
|-----------|--------|----------------------|-------------|
| `n_estimators` | 100 | 10–1000+ | Number of trees in the forest. More trees → better accuracy, slower training. |
| `criterion` | "gini" | "gini", "entropy", "log_loss" | Function to measure split quality. |
| `max_depth` | None | 2–30 | Maximum depth of each tree. None = fully grown until leaves are pure or min samples split reached. |
| `min_samples_split` | 2 | 2–20 | Minimum number of samples required to split an internal node (int or fraction). |
| `min_samples_leaf` | 1 | 1–10 | Minimum number of samples required at a leaf node. |
| `max_features` | "sqrt" | int, float, "sqrt", "log2", None | Number of features considered at each split. |
| `bootstrap` | True | True / False | Whether to use bootstrapped samples for training each tree. |
| `oob_score` | False | True / False | Whether to use out-of-bag samples to estimate generalization score. |
| `class_weight` | None | None, "balanced", "balanced_subsample", dict | Weights associated with classes. Adjusts for imbalanced datasets. |
| `random_state` | None | int | Controls randomness of bootstrapping and feature selection. |

> Other parameters like `max_leaf_nodes`, `min_impurity_decrease`, `ccp_alpha` control tree growth, pruning, and complexity.

---

## 3. **Important Attributes**

| Attribute | Description |
|-----------|-------------|
| `estimators_` | List of fitted `DecisionTreeClassifier` objects. |
| `classes_` | Class labels for single/multi-output problems. |
| `n_classes_` | Number of classes (single output) or list of numbers per output. |
| `feature_importances_` | Impurity-based feature importances. |
| `oob_score_` | Out-of-bag accuracy score (if `oob_score=True`). |
| `estimators_samples_` | Subset of samples used for each base estimator. |

---

## 4. **Key Methods**

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the random forest on data `X` with labels `y`. |
| `predict(X)` | Predict class labels for samples in `X` by majority vote. |
| `predict_proba(X)` | Predict class probabilities for each sample. |
| `predict_log_proba(X)` | Predict log of class probabilities. |
| `score(X, y)` | Compute accuracy of predictions. |
| `apply(X)` | Return the leaf index for each sample in each tree. |
| `decision_path(X)` | Return sparse matrix showing the nodes traversed for each sample. |

---

## 5. **How Random Forest Works**

1. Draw `n_estimators` **bootstrap samples** from the dataset.  
2. Train a **decision tree** on each sample, considering only `max_features` features at each split.  
3. Each tree predicts independently; the forest outputs the **majority vote** (classification).  
4. Optionally, use **OOB samples** to estimate model accuracy without a separate validation set.

---

## 6. **Real-World Examples**

| Scenario | Input Features | Target | Description |
|----------|----------------|--------|-------------|
| Email Filtering | Email text features (keywords, length, sender) | Spam / Not Spam | Classify incoming emails. |
| Medical Diagnosis | Blood tests, BMI, age, glucose level | Disease / No Disease | Predict presence of diabetes or heart disease. |
| Loan Approval | Income, credit score, employment status | Approved / Denied | Classify loan applications. |
| Customer Churn | Usage data, contract type, complaints | Churn / Not Churn | Predict if a customer will leave. |

---

## 7. **Example Code**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

# Create classifier
clf = RandomForestClassifier(max_depth=2, random_state=0)

# Train the model
clf.fit(X, y)

# Predict new sample
print(clf.predict([[0, 0, 0, 0]]))  # e.g., [1]
