# MAGIC Gamma Telescope – Machine Learning Classification

This project focuses on classifying **high-energy gamma-ray events** versus **hadronic background noise** using the **MAGIC Gamma Telescope dataset**. Multiple machine learning models are implemented and evaluated using standard classification metrics.

🔗 **Dataset Source:** UCI Machine Learning Repository  

---

## 📂 Dataset Overview

- **Source:** UCI Machine Learning Repository  
- **Number of Samples:** 19,020  
- **Number of Features:** 10 numerical attributes describing gamma-ray and hadron events  
- **Target Classes:**
  - `1 (g)` → Gamma-ray signal  
  - `0 (h)` → Hadronic background  

This dataset represents a real-world classification problem with non-linear patterns and class imbalance.

---

## 🧠 Machine Learning Models Used

The following classification algorithms were implemented and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Neural Network (Keras – Feed Forward)

---

## 📊 Model Performance Comparison

| Model               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|--------------------|----------|----------------------|------------------|--------------------|
| KNN                | 0.79     | 0.85                 | 0.83             | 0.84               |
| Naive Bayes        | 0.73     | 0.75                 | 0.88             | 0.81               |
| Logistic Regression| 0.77     | 0.83                 | 0.81             | 0.82               |
| SVM                | 0.85     | 0.88                 | 0.88             | 0.88               |
| Neural Network     | 0.87     | 0.88                 | 0.93             | 0.90               |

---

## 📈 Why SVM and Neural Networks Performed Better

### ✅ Support Vector Machine (SVM)
- Performs well on high-dimensional feature spaces.
- Effectively finds an optimal decision boundary with maximum margin.
- Benefited significantly from feature scaling and preprocessing.

### ✅ Neural Network
- Captured complex non-linear relationships between features.
- Oversampling helped address class imbalance and improved generalization.
- Feature normalization improved training stability and convergence.

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Label encoding of target variable  
- Feature standardization  
- Train / Validation / Test split (60% / 20% / 20%)

### 2️⃣ Handling Class Imbalance
- Random oversampling applied **only on training data**

### 3️⃣ Model Training
- Each model trained and validated independently
- Hyperparameters selected manually for simplicity

### 4️⃣ Evaluation
- Metrics used: Precision, Recall, F1-score, Accuracy
- Feature-wise histogram plots for class-wise distribution analysis

---

## 📁 Project Structure

- `MAGIC_GammaTelescope_ML` → Complete implementation (preprocessing, training, evaluation)
- `README.md` → Project documentation

---

## 🧠 Future Improvements

- Hyperparameter tuning using `GridSearchCV` or `Optuna`
- Deeper neural networks with dropout regularization
- Dimensionality reduction using PCA
- Advanced visualizations using t-SNE or UMAP

---

## 🤝 Acknowledgements

- MAGIC Collaboration  
- UCI Machine Learning Repository  
- Scikit-learn and Keras libraries  

---
