# Diabetic

# 🧠 Early Diabetes Diagnosis Using a Low-Complexity Deep Learning Model with Feature Reduction

**Author**: [Soroush Soltanizadeh](https://www.linkedin.com/in/soroush-soltanizadeh-1136892b6/)
**Google Scholar**: [Profile](https://scholar.google.com/citations?user=ARKNJYwAAAAJ&hl=en)

---

## 📍 Project Overview

Diabetes is a chronic condition that, if left undiagnosed or untreated, can lead to serious health complications. Early diagnosis is essential for timely intervention and treatment. This project presents a **low-complexity deep learning approach** to detect diabetes at early stages using the **Pima Indians Diabetes Database (PIDD)**. The study integrates **feature reduction techniques** with a **1-D Convolutional Neural Network (CNN)** to optimize both **diagnostic performance** and **computational efficiency**.

---

## 🎯 Objectives

* Build an efficient CNN model for early diabetes prediction
* Reduce input dimensionality using **Recursive Feature Elimination (RFE)**
* Analyze the contribution of individual clinical features
* Evaluate model robustness using key metrics and **SHAP** analysis for interpretability

---

## 📊 Dataset

* **Source**: Pima Indians Diabetes Database (PIDD)
* **Features**: Clinical attributes such as age, BMI, blood pressure, glucose, insulin, etc.
* **Target**: Diabetes diagnosis (`1`: Positive, `0`: Negative)

---

## 🛠️ Methodology

### ✅ Data Preprocessing

* Imputation of missing values using mean strategy
* Standardization using `StandardScaler`
* Feature selection via **RFE** with Logistic Regression

### 🧠 Model Architecture

* **1D-CNN** with:

  * Conv1D layer
  * MaxPooling
  * Dense output layer with sigmoid activation
* Optimized using **Adam optimizer** and **binary cross-entropy loss**

### 🧪 Evaluation Metrics

* Accuracy, F1-score, Precision
* Mean Squared Error (MSE)
* CNN Model Complexity (number of computations)
* **SHAP** analysis for feature importance

---

## 📈 Results

| Metric             | Value                                                                         |
| ------------------ | ----------------------------------------------------------------------------- |
| **Train Accuracy** | 97.20%                                                                        |
| **Test Accuracy**  | 96.52%                                                                        |
| **F1 Score**       | 98%                                                                           |
| **Precision**      | 100%                                                                          |
| **MSE**            | \~0.025                                                                       |
| **CNN Complexity** | Computed as `ni * nf * nk * (ns + 2*npad - dilation*(nk-1) - 1 / stride + 1)` |

> 💡 The CNN model demonstrates high accuracy with minimal complexity, making it ideal for real-time diagnosis applications in clinical settings.

---

## 🔍 Feature Importance

### 🔑 Selected Features:

* Glucose
* BMI
* Age
* BloodPressure
* Insulin
* SkinThickness
* Pregnancies

These features were identified by **RFE** as the most predictive variables and confirmed through **SHAP analysis**, which explained their influence on the model’s decisions.

---

## 📌 Key Takeaways

* **Low-Complexity CNN Model** provides excellent predictive performance for diabetes diagnosis.
* **Feature Reduction (RFE)** improves model efficiency without sacrificing accuracy.
* **SHAP** interpretability supports clinical decision-making by identifying key contributing features.
* The approach is robust, generalizes well, and is suitable for deployment in real-world healthcare systems.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas scikit-learn
```

### Run the Model

```bash
python diabetes_cnn_model.py
```

---

## 📚 Future Work

* Integrate SHAP visualizations for dynamic analysis
* Extend model to multi-disease diagnosis
* Deploy via a user-friendly interface for healthcare professionals

---

## 📝 License

This project is licensed under the **MIT License** – feel free to use and modify.
