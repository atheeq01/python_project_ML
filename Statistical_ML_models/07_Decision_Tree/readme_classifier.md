# Decision Tree Classifier – Complete Guide

---

## 1. What is DecisionTreeClassifier?

`DecisionTreeClassifier` is a **supervised learning algorithm** used for **classification tasks**.  

- It splits the dataset into **nodes** based on feature values to predict classes.  
- Works well for both **numerical and categorical data**.  
- Can capture **non-linear relationships** in the data.  
- Prone to **overfitting** if not properly controlled (max_depth, min_samples_leaf, etc.).

**Tip:** Use **RandomForestClassifier** or **GradientBoosting** for better generalization.

---

## 2. Key Parameters (with real-world examples and ranges)

| Parameter                 | Meaning                                                                                     | Real-World Example                                         | Typical Range / Values                         |
|---------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------|
| **criterion**             | Function to measure quality of a split                                                      | `gini` for customer churn prediction                       | `'gini'`, `'entropy'`, `'log_loss'`          |
| **splitter**              | Strategy to choose split at each node                                                       | `best` for optimal split, `random` for faster approximation| `'best'`, `'random'`                          |
| **max_depth**             | Maximum depth of the tree                                                                  | Limit complexity for loan default prediction               | int ≥1 or `None` (unlimited)                 |
| **min_samples_split**     | Minimum samples required to split a node                                                  | 5 samples required to split a node in medical diagnosis    | int ≥2 or float (fraction)                   |
| **min_samples_leaf**      | Minimum samples required at a leaf node                                                  | At least 3 patients in a terminal node                     | int ≥1 or float (fraction)                   |
| **min_weight_fraction_leaf** | Minimum weighted fraction at a leaf                                                       | Weighted data for imbalanced classes                       | float, default=0.0                             |
| **max_features**          | Number of features to consider for best split                                            | Random subset for faster splitting                          | int, float, `'sqrt'`, `'log2'`, `None`      |
| **random_state**          | Seed for reproducibility                                                                   | Reproducible train/test split                                | int or `None`                                 |
| **max_leaf_nodes**        | Maximum number of leaf nodes                                                              | Limit tree size for interpretability                        | int ≥2 or `None`                              |
| **min_impurity_decrease** | Minimum decrease in impurity required to split a node                                     | Prevent tiny splits in noisy data                            | float ≥0.0                                    |
| **class_weight**          | Weights associated with classes                                                          | Imbalanced fraud detection                                   | dict, `'balanced'`, `None`                   |
| **ccp_alpha**             | Complexity parameter for pruning (Minimal Cost-Complexity Pruning)                        | Post-prune branches in overfitting tree                     | float ≥0.0                                    |
| **monotonic_cst**         | Enforce monotonic increase/decrease constraint per feature                                | Increasing age should not reduce credit risk                | array-like of int [-1,0,1]                    |

---

## 3. Important Attributes After Fitting

| Attribute                 | Meaning                                                                                     |
|---------------------------|---------------------------------------------------------------------------------------------|
| `classes_`                | Class labels                                                                                 |
| `feature_importances_`    | Importance of each feature in prediction                                                    |
| `max_features_`           | Actual number of features considered during splitting                                      |
| `n_classes_`              | Number of classes                                                                         |
| `n_features_in_`          | Number of features seen during fit                                                         |
| `feature_names_in_`       | Names of features seen during fit                                                          |
| `n_outputs_`              | Number of outputs                                                                          |
| `tree_`                   | Underlying Tree object with node and split details                                         |

---

## 4. Key Methods

| Method                          | Description                                                   |
|--------------------------------|---------------------------------------------------------------|
| `fit(X, y)`                     | Train the decision tree on data `X` with labels `y`           |
| `predict(X)`                     | Predict class labels for samples in `X`                       |
| `predict_proba(X)`               | Predict probability for each class                             |
| `predict_log_proba(X)`           | Predict log-probabilities for each class                       |
| `score(X, y)`                    | Compute accuracy of the model on data `X`                     |
| `get_depth()`                     | Return maximum depth of the tree                               |
| `get_n_leaves()`                  | Return number of leaves in the tree                            |
| `apply(X)`                        | Return leaf indices where each sample ends up                 |
| `decision_path(X)`                | Return sparse matrix showing decision path for each sample    |
| `cost_complexity_pruning_path(X, y)` | Compute pruning path for Minimal Cost-Complexity Pruning     |
| `get_params()` / `set_params()`   | Get or set model parameters                                    |

---

## 5. Tips

1. **Scale features** only if using distance-based splits (not required for Decision Trees).  
2. **Limit tree size** using `max_depth`, `min_samples_leaf`, or `max_leaf_nodes` to avoid overfitting.  
3. **Class imbalance** → use `class_weight='balanced'`.  
4. **Post-pruning** → use `ccp_alpha` to reduce tree complexity.  
5. **Random splits** → `splitter='random'` can help speed up large datasets.  

---

## 6. Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# Train and evaluate
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", scores)

# Fit and predict
clf.fit(X, y)
print("Predicted class:", clf.predict([[5.0, 3.4, 1.5, 0.2]]))
print("Predicted probabilities:", clf.predict_proba([[5.0, 3.4, 1.5, 0.2]]))
