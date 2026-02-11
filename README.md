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

