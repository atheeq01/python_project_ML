<div style="margin-left: 20px; margin-right: 20px;">

# 📘 **KNeighborsClassifier – Parameters Explained (with Real-World Examples)**

```python
from sklearn.neighbors import KNeighborsClassifier
```

---

# 1. **n_neighbors**

### ✔ What it means

Number of nearest neighbors (K) used to classify a new data point.

### ✔ Why use it

Controls how many neighbors vote for the final class.

* Small K → more flexible, may overfit
* Large K → more stable, may underfit

### ✔ Real-world Example

Classifying whether a fruit is **apple or orange** based on its weight + color.

* With **K=1**, the closest fruit decides the class → very sensitive
* With **K=10**, decision is smoother

### ✔ Typical Range / Values

```
1–50
(Most common: 3, 5, 7, 9)
```

---

# 2. **weights**

### ✔ What it means

How the influence of neighbors is calculated.

Values:

* **"uniform"** → all neighbors have equal vote
* **"distance"** → closer neighbors have stronger vote
* **callable** → user-defined weighting function

### ✔ Why use it

Improves accuracy when data points closer to the target matter more.

### ✔ Real-world Example

Predicting if a credit card transaction is **fraudulent**.

* A very close similar transaction is more important → `weights="distance"`

### ✔ Typical Values

```
"uniform" (default)
"distance"
custom function
```

---

# 3. **algorithm**

### ✔ What it means

Which algorithm is used to find nearest neighbors.

Options:

* **"auto"** – chooses best automatically
* **"ball_tree"** – good for high dimensional
* **"kd_tree"** – good for medium dimensional
* **"brute"** – slow but simple (distance to all points)

### ✔ Why use it

Improves search speed for large datasets.

### ✔ Real-world Example

Face recognition system:

* High dimensions (128–2048 embedding vector)
* **ball_tree** or **brute** works better

### ✔ Typical Range / Values

```
"auto" (most common)
"brute" (for high dimensions)
"ball_tree" 
"kd_tree"
```

---

# 4. **leaf_size**

### ✔ What it means

Affects the speed/memory for BallTree / KDTree operations.

### ✔ Why use it

Smaller leaf size = deeper tree (slower query)
Larger leaf size = shallower tree (faster query)

### ✔ Real-world Example

Large e-commerce dataset recommending similar products:

* Increasing leaf size improves performance for millions of products

### ✔ Typical Range

```
20–100
(Default = 30)
```

---

# 5. **p (Minkowski metric power)**

### ✔ What it means

Defines the distance measure:

* **p = 1** → Manhattan distance
* **p = 2** → Euclidean distance (most used)
* **p > 2** → increasingly large penalty for bigger differences

### ✔ Why use it

Choosing the right distance metric can drastically improve accuracy.

### ✔ Real-world Example

Recommender system comparing user rating patterns:

* Manhattan distance (p=1) often performs better when user vectors are sparse.

### ✔ Typical Values

```
1 (L1)
2 (L2) – default
3–5 (rare)
```

---

# 6. **metric**

### ✔ What it means

Which distance formula to use (string or function).

Common values:

* **"minkowski"** → controlled by p
* **"euclidean"**
* **"manhattan"**
* **"cosine"**
* **"hamming"**

### ✔ Why use it

Different data types require different distance metrics.

### ✔ Real-world Example

Text similarity (Bag-of-Words vectors):

* Cosine distance performs better → `metric="cosine"`

### ✔ Typical Values

```
"minkowski" (default)
"euclidean"
"manhattan"
"cosine"
"hamming"
custom function
```

---

# 7. **metric_params**

### ✔ What it means

Additional arguments for custom distance metrics.

### ✔ Why use it

Allows fine-tuning the distance function.

### ✔ Real-world Example

When using a custom metric that needs extra parameters, such as:

* Scale factors
* Thresholds
* Penalties

### ✔ Typical Values

```
None (most common)
{"w": [0.2, 0.5, 1.0]}   # example custom parameters
```

---

# 8. **n_jobs**

### ✔ What it means

Number of CPU cores to use in neighbor search.

* `None` → 1 core
* `-1` → all available cores

### ✔ Why use it

Speeds up prediction, especially for large datasets.

### ✔ Real-world Example

Medical diagnostic system:

* 300k patient records
* Setting `n_jobs = -1` reduces prediction time dramatically

### ✔ Typical Values

```
None  (default)
1–8   (manual control)
-1    (use all cores)
```

---

# 📌 Summary Table

| Parameter         | Meaning                | Reason to Use         | Real-World Example        | Typical Range         |
| ----------------- | ---------------------- | --------------------- | ------------------------- | --------------------- |
| **n_neighbors**   | No. of neighbors       | Bias–variance control | Classify fruit type       | 3–15                  |
| **weights**       | Vote weighting         | Closer points matter  | Fraud detection           | "uniform", "distance" |
| **algorithm**     | Neighbor search method | Speed & efficiency    | Face recognition          | "auto", "brute"       |
| **leaf_size**     | Tree speed/memory      | Optimize search time  | Product recommendation    | 20–100                |
| **p**             | Distance power         | Choose L1/L2 metric   | User rating similarity    | 1, 2                  |
| **metric**        | Distance function      | Adapt to data type    | Text similarity           | "euclidean", "cosine" |
| **metric_params** | Extra params           | Custom metrics        | Weighted custom distances | None / dict           |
| **n_jobs**        | CPU cores              | Speed up computation  | Large medical dataset     | -1, None              |

---
</div>