# 🚀 E-commerce Customer Spending Analysis

Predict the **Yearly Amount Spent** by ecommerce customers using a **Linear Regression model** based on customer behavior on mobile app, website, and membership duration.

---

![Python](https://img.shields.io/badge/Python-3.10.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-✅-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-✅-green)

---

## 📋 Project Overview

This project analyzes customer behavior data from an e-commerce company to predict yearly spending amounts using linear regression.  
It includes data visualization, model training, evaluation, residual analysis, and saving the model for future predictions.

---

## 📊 Dataset Information

- Source: [Kaggle](https://www.kaggle.com/kolawale/focusing-on-mobile-app-or-website)  
- CSV file: `Ecommerce Customers.csv`  

| Feature | Description |
|---------|-------------|
| Avg. Session Length | Average session length (minutes) |
| Time on App | Time spent on mobile app (minutes) |
| Time on Website | Time spent on website (minutes) |
| Length of Membership | Membership duration (years) |
| Yearly Amount Spent | Target variable (dollars) |

**Explanation:**  
The goal is to predict yearly spending based on customer interaction with the app/website and how long they have been members.

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/atheeq01/python_project_ML/tree/master/linear_regression
cd <repository-directory>
pip install -r requirements.txt
```

## **📦 Dependencies** 
* Python 3.10
* pandas 2.3.2
* numpy 2.2.6
* matplotlib 3.10.6
* seaborn 0.13.2
* scikit-learn 1.7.2
* scipy 1.15.3
* joblib 1.5.2

---

## 🚀 How to Run

Run the Jupyter notebook `ecommerce_analysis.ipynb`. It will:

1. Download the dataset from Kaggle.
2. Perform Exploratory Data Analysis (EDA).
3. Train a Linear Regression model.
4. Evaluate model performance.
5. Save the trained model.

---

## 🛠 Project Steps with Explanations

### 1️⃣ Load Libraries & Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("assets/Ecommerce Customers.csv")
df.head()
```

*Explanation:*  
Load libraries for data analysis (`pandas`, `numpy`) and visualization (`matplotlib`, `seaborn`).  
`df.head()` shows the first rows to verify dataset structure.

---

### 2️⃣ Exploratory Data Analysis (EDA)

```python
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.pairplot(df, kind="scatter", plot_kws={"alpha":0.5})
sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df,
           scatter_kws={"alpha":0.3}, line_kws={"color":"red","linestyle":"--"})
plt.show()
```

*Explanation:*  
Visualizations help identify patterns and correlations.  
- Strong positive correlation: `Length of Membership` → important predictor.  
- Moderate correlation: `Time on App`.  
- Weak correlation: `Time on Website`.

---

### 3️⃣ Split Data into Training & Test Sets

```python
from sklearn.model_selection import train_test_split

X = df[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]
y = df["Yearly Amount Spent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

*Explanation:*  
- Separate features (`X`) and target (`y`).  
- Train on 70% of data, test on 30% → ensures fair evaluation.

---

### 4️⃣ Train Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

coef_df = pd.DataFrame(lm.coef_, X.columns, columns=["Coefficient"])
print(coef_df)
print("Intercept:", lm.intercept_)
```

*Explanation:*  
- `LinearRegression()` creates the model.  
- `fit()` trains it.  
- Coefficients indicate feature importance; intercept is the baseline.

---

### 5️⃣ Make Predictions

```python
y_pred = lm.predict(X_test)
```

*Explanation:*  
Use the trained model to predict yearly spending for test data.

---

### 6️⃣ Evaluate Model Performance

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

*Explanation:*  
- **MAE:** Average error in dollars → easy to interpret.  
- **MSE:** Penalizes larger errors → useful for optimization.  
- **RMSE:** Same units as target → interpretable.  
- Low errors (~$7–$10) relative to spending range (300–700) → model is good.

---

### 7️⃣ Residual Analysis

```python
residuals = y_test - y_pred

sns.displot(residuals, bins=20, kde=True)

import pylab, scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()
```

*Explanation:*  
- Residuals = actual – predicted.  
- Random, normal distribution → assumptions of linear regression satisfied.

---

### 8️⃣ Visualize Predictions vs Actual

```python
sns.scatterplot(x=y_pred, y=y_test)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual Yearly Amount Spent")
plt.show()
```

*Explanation:*  
- Points near the diagonal → accurate predictions.  
- Confirms model reliability visually.

---

### 9️⃣ Save Trained Model

```python
import joblib

joblib.dump(lm, "assets/linear_regression_model.pkl")
```

*Explanation:*  
- Save model to disk for reuse.  
- Load later for new predictions:

```python
lm = joblib.load("./assets/linear_regression_model.pkl")
new_customer = [[34.5, 12.65, 39.57, 4.08]]
predicted_spent = lm.predict(new_customer)
print("Predicted Yearly Amount Spent:", predicted_spent)
```

---

## 📈 Key Findings

### Model Coefficients
| Feature | Coefficient | Impact |
|---------|-------------|--------|
| Avg. Session Length | 25.98 | Positive |
| Time on App | 38.59 | Strong positive |
| Time on Website | 0.19 | Minimal |
| Length of Membership | 61.27 | Strongest positive |

### Model Performance
- **Mean Absolute Error (MAE):** $7.23  
- **Mean Squared Error (MSE):** $79.81  
- **Root Mean Squared Error (RMSE):** $8.93  

**Business Implications:**  
1. Invest in the mobile app → increases customer spending.  
2. Website improvements have minimal effect.  
3. Reward long-term members → strongest impact on spending.

---

## 📁 Project Structure

```
├── assets/
│   ├── Ecommerce Customers.csv
│   └── linear_regression_model.pkl
├── ecommerce_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- Test advanced regression models (Ridge, Lasso).  
- Perform clustering for customer segmentation.  
- Develop a web app for predictions.  
- Conduct A/B testing to improve revenue.

---

## 👥 Contributors

[Your Name]

---

## 🙏 Acknowledgments

- Dataset by Kolawale on Kaggle  
- Built with Python 3.10, pandas 2.3.2, numpy 2.2.6, matplotlib 3.10.6, seaborn 0.13.2, scikit-learn 1.7.2, scipy 1.15.3, joblib 1.5.2  
- Repository: [GitHub Link](https://github.com/atheeq01/python_project_ML/tree/master/linear_regression)

---

**Made with ❤️ using Python & scikit-learn**
