# Support Vector Classifier (SVC) – Complete Guide

---

## 1. What is SVC?

`SVC` stands for **C-Support Vector Classification**, part of **Support Vector Machines (SVM)** for classification tasks.

- Finds a **hyperplane** that best separates different classes in your data.
- Supports **non-linear classification** using **kernels** like `linear`, `poly`, `rbf`, `sigmoid`.
- **Based on `libsvm`**:
  - Works well for **small-to-medium datasets**.
  - Training time grows **quadratically** with the number of samples → impractical for very large datasets.

**Tip:** For large datasets, use `LinearSVC` or `SGDClassifier` with kernel approximation.

---

## 2. Key Parameters (with real-world examples and ranges)

| Parameter                 | Meaning                                                                                       | Real-World Example                                      | Typical Range / Values                     |
|---------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------|-------------------------------------------|
| **C**                     | Regularization strength. High → strict, Low → allows misclassifications                        | Email spam detection                                   | Positive float, e.g., 0.1, 1, 10, 100   |
| **kernel**                | Type of decision boundary                                                                     | Customer segmentation: linear/simple, rbf/complex     | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| **degree**                | Degree of polynomial kernel (`poly`)                                                           | Sales prediction curves                                 | Integer ≥ 0, default=3                    |
| **gamma**                 | Influence of a single point (rbf/poly/sigmoid)                                               | Fraud detection: high = local, low = global           | `'scale'`, `'auto'`, or positive float   |
| **coef0**                 | Independent term for `poly` and `sigmoid` kernels                                             | Weight feature interactions in poly model             | Float, default=0.0                        |
| **shrinking**             | Use shrinking heuristic to speed up training                                                  | Large datasets                                        | `True` / `False` (default=True)          |
| **probability**           | Enable probability estimates (`predict_proba`)                                                | Loan default probability                               | `True` / `False` (default=False)         |
| **class_weight**          | Handle imbalanced classes                                                                    | Rare disease detection                                  | `None`, `'balanced'`, or dict            |
| **max_iter**              | Max iterations for solver                                                                     | Limit training on big datasets                         | Integer, `-1` = no limit                 |
| **decision_function_shape** | Multi-class strategy                                                                        | Digit recognition (0-9)                                | `'ovr'` (one-vs-rest), `'ovo'`           |
| **break_ties**            | Break ties using confidence values                                                           | Spam vs important email                                 | `True` / `False` (default=False)         |
| **random_state**          | Ensure reproducible results                                                                   | Experiment reproducibility                               | Integer or `None`                        |

---

## 3. Important Attributes After Fitting

| Attribute          | Meaning                                                       |
|------------------|---------------------------------------------------------------|
| `support_vectors_` | The vectors that define the hyperplane.                       |
| `n_support_`       | Number of support vectors per class.                          |
| `dual_coef_`       | Coefficients for support vectors in the decision function.    |
| `coef_`            | Weight vector (only for linear kernel).                       |
| `intercept_`       | Bias term in the hyperplane equation.                         |
| `classes_`         | Unique class labels.                                          |
| `probA_`, `probB_` | Parameters for probability estimates (`probability=True`).    |
| `n_iter_`          | Number of iterations taken to fit each model.                 |

---

## 4. Key Methods

| Method                          | Description                                                   |
|--------------------------------|---------------------------------------------------------------|
| `fit(X, y)`                     | Train the SVM model on data `X` with labels `y`.             |
| `predict(X)`                     | Predict class labels for samples in `X`.                     |
| `predict_proba(X)`               | Predict probability of each class (`probability=True`).      |
| `predict_log_proba(X)`           | Predict log probabilities of classes.                        |
| `decision_function(X)`           | Returns distance of samples to the separating hyperplane.    |
| `score(X, y)`                    | Returns accuracy of the model.                                |
| `get_params()` / `set_params()`  | Get or set model parameters (useful in pipelines/grid search)|

---

## 5. Notes on Multi-class Classification

- Internally, `SVC` always uses **one-vs-one (ovo)** to train multi-class models.
- `decision_function_shape='ovr'` converts the internal ovo output to **one-vs-rest**.

---

## 6. Tips

1. **Scale features** with `StandardScaler`.
2. For **large datasets**, use `LinearSVC` to save time.
3. Tune **C and gamma** with `GridSearchCV` for best performance.
4. **Kernel choice**:
   - `linear`: fast, linearly separable data
   - `rbf`: flexible, default non-linear
   - `poly`: polynomial boundaries
   - `sigmoid`: rarely used

---

## 7. Example

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale', probability=True))
clf.fit(X, y)

# Predict class
print(clf.predict([[-0.8, -1]]))  

# Predict probability
print(clf.predict_proba([[-0.8, -1]]))  
