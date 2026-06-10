# 💎 Diamond Price Predictor

A Machine Learning project that predicts diamond prices using physical and quality-related attributes such as carat, cut, color, clarity, depth, table, and dimensions. The project benchmarks multiple machine learning models and provides an interactive Streamlit dashboard for real-time price prediction.
<img width="1919" height="940" alt="image" src="https://github.com/user-attachments/assets/e97b5eeb-f92e-47a5-b237-f849c688caa8" />
<img width="1917" height="948" alt="image" src="https://github.com/user-attachments/assets/c7063f25-6cf4-420f-a008-cfca9b07db51" />
<img width="1919" height="957" alt="image" src="https://github.com/user-attachments/assets/50e84a7f-db2f-4271-a377-c0d4c2dce118" />



---

## 📌 Overview

Diamond pricing depends on several factors including size, cut quality, color, clarity, and physical dimensions. This project explores how different machine learning algorithms perform on this regression problem and identifies the best-performing model for price prediction.

In addition to model benchmarking, a Streamlit dashboard was developed to allow users to estimate diamond prices through an intuitive interface.

---

## 🚀 Features

* Data preprocessing and cleaning
* Missing value handling
* Outlier detection and removal
* Feature encoding and scaling
* Multiple machine learning models
* Model performance comparison
* Interactive Streamlit dashboard
* Real-time diamond price prediction

---

## 📊 Dataset

The project uses the Diamonds Dataset containing information such as:

* Carat
* Cut
* Color
* Clarity
* Depth
* Table
* X Dimension
* Y Dimension
* Z Dimension
* Price

Dataset Size:

* 53,940 diamond records

---

## 🤖 Machine Learning Models

### Regression Models

* Linear Regression
* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)
* XGBoost Regressor
* Multi-Layer Perceptron (MLP) Regressor

### Classification Models

To benchmark performance using categorized price ranges:

* Logistic Regression
* KNN Classifier
* Naive Bayes
* Decision Tree Classifier
* Random Forest Classifier
* SVM Classifier
* XGBoost Classifier
* Multi-Layer Perceptron (MLP) Classifier

---

## 📈 Model Evaluation

### Regression Metrics

* R² Score
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## 🖥️ Dashboard Preview

### Home Screen

Add screenshot here:

```text
screenshots/dashboard_home.png
```

### Prediction Example

Add screenshot here:

```text
screenshots/prediction_result.png
```

---

## 📂 Project Structure

```text
Diamond-Price-Predictor/

├── data/
│   └── diamonds.csv
│
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
│
├── app/
│   └── streamlit_app.py
│
├── preprocess.py
├── price_predictor.py
├── model_comparison.py
├── main.py
│
├── screenshots/
│   ├── dashboard_home.png
│   └── prediction_result.png
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Matplotlib

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Model Benchmarking

```bash
python main.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🎯 Key Learnings

Through this project I gained hands-on experience with:

* Data preprocessing techniques
* Feature engineering
* Regression and classification algorithms
* Model evaluation and comparison
* Neural Networks using Scikit-Learn
* Building interactive ML applications with Streamlit

---

## 🔮 Future Improvements

* Model persistence using Joblib/Pickle
* Hyperparameter tuning
* Cloud deployment
* Docker support
* Automated retraining pipeline
* Enhanced dashboard visualizations

---

## 👨‍💻 Author

**Yash Gupta**

B.Tech Electrical Engineering
National Institute of Technology Delhi

Areas of Interest:

* Machine Learning
* Artificial Intelligence
* Software Development
* Data Structures & Algorithms
