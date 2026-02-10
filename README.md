# Wine-Quality-ML-Project
Binary classification of wine quality using machine learning
# 🍷 Wine Quality Machine Learning Project

This repository contains the initial phase of a machine learning project using the Wine Quality dataset. **Phase 1 focuses only on environment setup, dataset upload, and basic data inspection.**

---

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
