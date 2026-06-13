# 💎 Diamond Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=flat-square&logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-53%2C940%20records-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

An end-to-end machine learning project that benchmarks **15 regression and classification models** for diamond price prediction, achieving **R² = 0.9811** with XGBoost and **95.98% classification accuracy** with Random Forest. Includes a live Streamlit dashboard for real-time price estimation.

---

## 🚀 Live Demo

> Try the Streamlit app locally — enter diamond attributes and get an instant price prediction.

![Dashboard Preview](https://github.com/user-attachments/assets/e97b5eeb-f92e-47a5-b237-f849c688caa8)

---

## 📈 Results

### Regression (Best Models)

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **XGBoost** | **0.9811** | **0.1359** | **0.0753** |
| Random Forest | 0.9802 | 0.1392 | 0.0747 |
| KNN Regressor | 0.9644 | 0.1867 | 0.1082 |
| Decision Tree | 0.9642 | 0.1872 | 0.0991 |
| Linear Regression | 0.8915 | 0.3258 | 0.2208 |

### Classification (Price Ranges)

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **95.98%** |
| XGBoost Classifier | ~95% |
| MLP Classifier | ~93% |

---

## 🧠 How It Works

```
Raw Data (53,940 records)
        │
        ▼
Preprocessing & Cleaning
(missing values, outliers, encoding, scaling)
        │
        ▼
Feature Engineering
(carat, cut, color, clarity, depth, table, x, y, z)
        │
   ┌────┴────┐
   ▼         ▼
Regression   Classification
(predict     (predict price
price $)     range bucket)
   │         │
   └────┬────┘
        ▼
Model Benchmarking (15 models)
        │
        ▼
Best Model → Streamlit Dashboard
```

---

## 🤖 Models Benchmarked

**Regression (7 models)**
Linear Regression · KNN Regressor · Decision Tree · Random Forest · SVR · XGBoost · MLP Regressor

**Classification (8 models)**
Logistic Regression · KNN · Naive Bayes · Decision Tree · Random Forest · SVM · XGBoost · MLP Classifier

---

## 📊 Dataset

[Diamonds Dataset — Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds)

| Property | Value |
|----------|-------|
| Records | 53,940 |
| Features | 9 (carat, cut, color, clarity, depth, table, x, y, z) |
| Target | Price (USD) |
| Size | ~3.4 MB |

---

## 🖥️ Dashboard

![Prediction Example](https://github.com/user-attachments/assets/c7063f25-6cf4-420f-a008-cfca9b07db51)
![Model Comparison](https://github.com/user-attachments/assets/50e84a7f-db2f-4271-a377-c0d4c2dce118)

---

## 📁 Project Structure

```
Diamond-Price-Predictor/
├── data/
│   └── diamonds.csv
├── models/
│   ├── linear_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── knn.py
│   ├── svm.py
│   ├── xgboost_model.py
│   ├── neural_network.py
│   ├── logistic_regression.py
│   └── naive_bayes.py
├── app/
│   └── streamlit_app.py
├── preprocess.py
├── price_predictor.py
├── model_comparison.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YASH-GUPTA995/Diamond-Price-Prediction-And-Model-Benchmarking.git
cd Diamond-Price-Prediction-And-Model-Benchmarking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run model benchmarking
python main.py

# 4. Launch the Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Scikit-Learn` | 13 of the 15 ML models, preprocessing |
| `XGBoost` | Best-performing regression + classification |
| `Pandas / NumPy` | Data manipulation and feature engineering |
| `Streamlit` | Interactive prediction dashboard |
| `Matplotlib` | Model comparison visualizations |

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Model persistence with Joblib
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
- [ ] SHAP values for feature importance explainability
- [ ] Docker support

---

## 👨‍💻 Author

**Yash Gupta** · B.Tech Electrical Engineering · NIT Delhi

[GitHub](https://github.com/YASH-GUPTA995) · [LinkedIn](https://www.linkedin.com/in/yash-gupta-nit-delhi/)
