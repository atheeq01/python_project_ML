<div style="margin-left: 20px; margin-right: 20px;">

---

# 📌 **KNeighborsRegressor — Parameters Explained (with Examples & Ranges)**

`KNeighborsRegressor` predicts a numeric value by looking at the **closest K samples** in the training data and averaging/weighting their values.

---

## ## 1. **n_neighbors (K value)**

**Meaning:**
How many nearest neighbors should influence the prediction.

**Why use it:**
Controls model complexity:

* Low K → model becomes noisy, overfits
* High K → smoother predictions, may underfit

**Typical range:**

* Small datasets: **3 – 10**
* Medium datasets: **10 – 50**
* Large datasets: **50 – 200+**
* Must be ≥1

**Real-world example:**
Predicting **house price** — using

* 5 nearest similar houses → more sensitive
* 50 nearest houses → more stable

```python
KNeighborsRegressor(n_neighbors=5)
```

---

## ## 2. **weights (‘uniform’ or ‘distance’)**

### **weights = "uniform"**

Every neighbor contributes equally.

👉 Best when all neighbors are similarly important.

### **weights = "distance"**

Closer neighbors get higher influence (1/distance).

👉 Best when closer points matter more.

### **weights = callable**

Custom weight function.

**Typical values:**

* `"uniform"`
* `"distance"`
* `lambda dist: 1/(dist+1)` (custom)

**Example (distance weights):**
Predicting **air pollution level** at a location — nearby stations should influence more.

```python
KNeighborsRegressor(weights='distance')
```

---

## ## 3. **algorithm ('auto', 'ball_tree', 'kd_tree', 'brute')**

**Meaning:**
How nearest neighbors are searched.

| Algorithm     | Best When                           | Notes                    |
| ------------- | ----------------------------------- | ------------------------ |
| **auto**      | Default for all                     | Automatically picks best |
| **kd_tree**   | Low dimensional data (<20 features) | Very fast                |
| **ball_tree** | High-dimensional data               | More flexible            |
| **brute**     | Very large data or sparse matrices  | Slowest                  |

**Typical values:**

* `"auto"` in almost all real-world tasks.

**Real-world example:**
Predicting **credit score** based on 50+ features → use `ball_tree`.

```python
KNeighborsRegressor(algorithm='auto')
```

---

## ## 4. **leaf_size (default=30)**

Used in `kd_tree` and `ball_tree`.

**Meaning:**
Controls number of points in each leaf of the tree.

**Why use it:**

* Low leaf_size → slower training, faster prediction
* High leaf_size → faster training, slower prediction

**Typical range:**

* **20 – 50**
* Rarely tuned unless dataset is huge

**Real-world example:**
Climate data with 500k+ points → increase leaf_size to 40–60.

---

## ## 5. **p (Power parameter for distance)**

Defines which Minkowski distance is used.

| p value | Distance type  | Usage                        |
| ------- | -------------- | ---------------------------- |
| **1**   | Manhattan (L1) | High-dimensional sparse data |
| **2**   | Euclidean (L2) | Most common choice           |
| **>2**  | Higher-order   | Rarely used                  |

**Typical range:**

* **p = 1 or 2**

**Real-world example:**
Predicting **delivery time** where features are grid-like (city blocks) → use `p=1`.

```python
KNeighborsRegressor(p=2)
```

---

## ## 6. **metric (‘minkowski’, ‘euclidean’, ‘manhattan’, etc.)**

**Meaning:**
Distance metric used between samples.

Common metrics:

| metric          | Equivalent to                        | When to use                 |
| --------------- | ------------------------------------ | --------------------------- |
| **'minkowski'** | p = 1 → manhattan, p = 2 → euclidean | Default                     |
| **'euclidean'** | L2                                   | Most general-purpose        |
| **'manhattan'** | L1                                   | Sparse/high-dimensional     |
| **'chebyshev'** | L∞                                   | Strict sensitivity problems |
| **'hamming'**   | Binary features                      | Text or yes/no features     |

**Typical range:**

* `"minkowski"` (default)
* `"euclidean"`
* `"manhattan"`

**Real-world example:**
Predicting **taxi trip duration** in a grid city → `"manhattan"` distance.

---

## ## 7. **metric_params (default=None)**

Extra parameters for the distance metric.

**When used:**
Rarely used by beginners.
Used for advanced custom distance calculations.

**Example:**
Custom scaling per feature.

---

## ## 8. **n_jobs**

Number of CPU cores to use.

| Value    | Meaning                  |
| -------- | ------------------------ |
| **None** | 1 core                   |
| **-1**   | use all CPU cores        |
| **>1**   | specific number of cores |

**Why use it:**
To speed up prediction on large datasets.

**Real-world example:**
Predicting **market prices** on 2 million samples → use `n_jobs=-1`.

---

# 📌 **Final Summary Table**

| Parameter         | Meaning                              | Why Use It                     | Typical Range                     | Real-World Example       |
| ----------------- | ------------------------------------ | ------------------------------ | --------------------------------- | ------------------------ |
| **n_neighbors**   | Number of neighbors                  | Controls smoothness            | 3–50                              | Predict house price      |
| **weights**       | Uniform or distance-based importance | Influence of closer points     | uniform / distance                | Air pollution prediction |
| **algorithm**     | Neighbor search strategy             | Speed optimization             | auto                              | Credit risk prediction   |
| **leaf_size**     | Tree leaf size                       | Balance between speed & memory | 20–50                             | Climate dataset          |
| **p**             | Distance power                       | Type of distance metric        | 1 or 2                            | Delivery time in cities  |
| **metric**        | Distance calculation formula         | Better similarity measurement  | minkowski / euclidean / manhattan | Market trend prediction  |
| **metric_params** | Extra metric config                  | Custom distances               | Usually None                      | Custom feature weighting |
| **n_jobs**        | Parallelization                      | Speed boost                    | -1 (all cores)                    | Big data regression      |

---

</div>