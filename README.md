# Wine-Quality-ML-Project
Binary classification of wine quality using machine learning
# 🍷 Wine Quality Machine Learning Project

This repository contains the initial phase of a machine learning project using the Wine Quality dataset. **Phase 1 focuses only on environment setup, dataset upload, and basic data inspection.**

---
## 🛠️ Tools Used
* Python
* Pandas
* NumPy
* Google Colab

## 📁 Repository Structure
Wine-Quality-ML-Project/
│
├── Wine_Quality_ML_Project.ipynb
└── README.md

## 📌 Phase 1: Dataset Loading & Inspection

### Step 1: Import Required Libraries
```bash
import pandas as pd
import numpy as np
```

### Step 2: Upload Dataset in Google Colab
```bash
from google.colab import files
uploaded = files.upload()
```
### Step 3: Load Dataset
```bash
df = pd.read_csv("winequality (1).csv")
```
### Step 4: Preview Dataset
```bash
df.head()
```
### Step 5: Check Dataset Information
```bash
df.info()
```
### Step 6: Check Dataset Shape
```bash
df.shape
```
---

## 📊 Dataset Details
* **Dataset:** Wine Quality (Red Wine)
* **Total Rows:** 1599
* **Total Columns:** 12
* **Target Column:** quality
* **Missing Values:** None


# 📌 Phase 2: Data Preprocessing

## 🔹 1. Check Missing Values

```python
df.isnull().sum()
```

✅ **Insight:**  
No missing values found in any column.

---

## 🔹 2. Check Duplicate Rows

```python
df.duplicated().sum()
```

📊 **Output:**  
240 duplicate rows found.

---

## 🔹 3. Remove Duplicate Rows

```python
df = df.drop_duplicates()
df.shape
```

📊 **Final Dataset Shape:**  
**1359 rows × 12 columns**

✅ **Insight:**  
Duplicate rows removed to prevent biased model learning.

---

## 🔹 4. Verify Data Types

```python
df.info()
```

✅ **Insight:**

- All features are `float64`
- Target (`quality`) is `int64`
- No encoding required

---

## 🔹 5. Convert Target to Binary Classification

### Project Requirement:

- `quality ≥ 7` → **1 (Good Wine)**
- `quality < 7` → **0 (Not Good Wine)**

```python
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df['quality_label'].value_counts()
```

📊 **Class Distribution:**

- Not Good (0): **1175**
- Good (1): **184**

⚠ **Insight:**  
Dataset is highly imbalanced (**86.5% vs 13.5%**).  
Accuracy alone is not a reliable evaluation metric.

---

# 📌 Phase 3: Exploratory Data Analysis (EDA)

## 🔹 1. Correlation Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

---

## 📊 Key Feature Insights

### 🔥 Strong Positive Impact on Wine Quality

| Feature        | Correlation with `quality_label` |
|---------------|----------------------------------|
| Alcohol       | 0.41 |
| Sulphates     | 0.20 |
| Citric Acid   | 0.20 |

✔ Higher alcohol content increases probability of good quality wine.

---

### ❌ Negative Impact on Quality

| Feature               | Correlation |
|-----------------------|-------------|
| Volatile Acidity      | -0.27 |
| Density               | -0.16 |
| Total Sulfur Dioxide  | -0.14 |

✔ Higher volatile acidity reduces wine quality.

---

## 🔹 2. Multicollinearity Check

Observed correlations between features:

- Free sulfur dioxide ↔ Total sulfur dioxide → **0.67**
- Fixed acidity ↔ pH → **-0.69**
- Alcohol ↔ Density → **-0.50**

✅ **Insight:**  
No feature correlation above **±0.85**.  
No severe multicollinearity issue detected.  
All features retained.

---

## 🔹 3. Class Distribution Visualization

```python
sns.countplot(x='quality_label', data=df)
plt.title("Binary Class Distribution")
plt.show()
```

⚠ **Insight:**  
Strong class imbalance detected.

Model evaluation must include:

- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 📊 Dataset Summary After Preprocessing

| Item | Status |
|------|--------|
| Missing Values | None |
| Duplicate Rows | 240 removed |
| Final Rows | 1359 |
| Features | 11 |
| Target Type | Binary Classification |
| Class Balance | Imbalanced |
# 📌 Phase 4: Feature Processing

## 🔹 1. Feature–Target Separation

After creating the binary target column `quality_label`, features and target were separated.

```python
X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]
```

✅ **Explanation**

- `X` contains 11 chemical features  
- `y` contains binary classification label (0 = Bad, 1 = Good)

---

## 🔹 2. Train-Test Split

The dataset was split into 80% training and 20% testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

✅ **Explanation**

- `random_state=42` ensures reproducibility  
- Testing data is used only for final evaluation  

---

## 🔹 3. Feature Scaling

Scaling is necessary for distance-based models like Logistic Regression and KNN.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

✅ **Explanation**

- Ensures all features have similar scale  
- Prevents large-magnitude features from dominating  
- Improves convergence and performance  

---

# 📌 Phase 5: Logistic Regression (Without Scaling)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
```

## 📊 Evaluation

```python
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
```

### 📊 Results

- Accuracy: **0.893**
- Precision: **0.667**
- Recall: **0.294**
- F1 Score: **0.408**

⚠ **Insight**

Although accuracy is high, recall is very low.  
The model struggles to detect GOOD wines due to dataset imbalance.

---

# 📌 Phase 6: Logistic Regression (With Scaling)

```python
lr_scaled = LogisticRegression(max_iter=1000)
lr_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = lr_scaled.predict(X_test_scaled)
```

### 📊 Results

- Accuracy: **0.893**
- Precision: **0.619**
- Recall: **0.382**
- F1 Score: **0.472**

✅ **Insight**

Scaling improved recall and F1-score, showing better detection of GOOD wines.

---

# 📌 Phase 7: K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
```

### 📊 Results

- Accuracy: **0.886**
- Precision: **0.552**
- Recall: **0.471**
- F1 Score: **0.508**

✅ **Insight**

KNN improved recall compared to Logistic Regression, making it better at identifying GOOD wines.

---

# 📌 Phase 8: Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
```

### 📊 Results

- Accuracy: **0.882**
- Precision: **0.528**
- Recall: **0.558**
- F1 Score: **0.543**

✅ **Insight**

Decision Tree performed better in terms of recall and F1-score, capturing non-linear feature relationships.

---

# 📌 Phase 9: Hyperparameter Tuning (GridSearchCV)

To improve Decision Tree performance:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
```

## 🔹 Best Parameters Found

- `max_depth = 7`
- `min_samples_split = 10`
- `min_samples_leaf = 1`

---

## 🔹 Tuned Model Evaluation

```python
y_best = best_model.predict(X_test)

accuracy_score(y_test, y_best)
precision_score(y_test, y_best)
recall_score(y_test, y_best)
f1_score(y_test, y_best)
confusion_matrix(y_test, y_best)
```

### 📊 Tuned Results

- Accuracy: **0.886**
- Precision: **0.541**
- Recall: **0.588**
- F1 Score: **0.563**

✅ **Insight**

Hyperparameter tuning improved recall and F1-score.  
Tuned Decision Tree became the best-performing model.

---

# 📌 Confusion Matrix Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_best),
            annot=True,
            fmt='d')
plt.title("Tuned Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Interpretation

- True Negatives: **221**
- True Positives: **20**
- False Positives: **17**
- False Negatives: **14**

The model correctly detects 20 GOOD wines while maintaining strong overall accuracy.

---

# 📌 Feature Importance Analysis

```python
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance
```

## 🔑 Key Influential Features

- Alcohol (Strong positive impact)
- Sulphates
- Volatile Acidity (Negative impact)

---

## ✅ Business Insight

- Higher alcohol content increases probability of GOOD quality.
- High volatile acidity reduces wine quality.
- Maintaining proper chemical balance improves ratings.

---

# 📌 Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression (No Scale) | 0.893 | 0.667 | 0.294 | 0.408 |
| Logistic Regression (Scaled) | 0.893 | 0.619 | 0.382 | 0.472 |
| KNN | 0.886 | 0.552 | 0.471 | 0.508 |
| Decision Tree | 0.882 | 0.528 | 0.558 | 0.543 |
| Tuned Decision Tree | 0.886 | 0.541 | 0.588 | 0.563 |

---

# 🏆 Final Conclusion

Among all models tested, the **Tuned Decision Tree achieved the highest Recall and F1-score**.

Since the dataset is imbalanced, **F1-score was selected as the primary evaluation metric**.

Therefore, the **Tuned Decision Tree** is selected as the final model for wine quality classification.

