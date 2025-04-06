# 🧠 Student Depression Analysis & Prediction

This repository presents a comprehensive **machine learning pipeline** to analyze and predict depression in students using a real-world dataset. It includes **data preprocessing**, **visualization**, **feature engineering**, **model training**, and **interactive prediction UI using Streamlit**.

---

## 📁 Project Structure

```
Student-Depression-Analysis-Prediction/
│
├── data/
│   └── Student Depression Dataset.csv     # Raw dataset used for training and analysis
│
├── app.py                                             # Main Streamlit dashboard with EDA + ML
├── run.ipynb                                          # Run this file for pickle model genration
├── student-depression-classification.ipynb            # Run this file for all the algorithm dependancies
├── README.md                                          # You're here!
```

---

## 🚀 Features

- 📊 **Interactive EDA**: Univariate & Bivariate analysis with Seaborn and Matplotlib
- 🧹 **Preprocessing**: Outlier treatment, missing value imputation, categorical encoding
- 🧠 **ML Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Voting Classifier (Ensemble)
- ✅ **Model Evaluation**:
  - Classification report
  - Confusion Matrix
  - ROC Curve (with AUC Scores)
- 🌐 **Streamlit App**: Clean interface to explore data and test different models

---

## 🛠️ Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/siddsahu17/Student-Depression-Analysis-Prediction.git
cd Student-Depression-Analysis-Prediction
```

2. **Install dependencies**:
Make sure you have Python 3.7+ and install packages:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install streamlit pandas numpy seaborn scikit-learn matplotlib xgboost lightgbm
```

3. **Run the Streamlit app**:
```bash
streamlit run app.py
```

---

## 📌 Notes

- The dataset contains demographic and lifestyle information, and a depression target.
- Feature engineering includes city normalization, sleep and diet ranking, etc.
- GridSearchCV is used for hyperparameter tuning on all models.

---

## 📈 Sample Screens

- Histogram & Count Plots
- Boxplots (Before/After Outlier Treatment)
- Confusion Matrix + ROC Curve Comparison
- Model Selector Dropdown for Evaluation

---

## 🤝 Contribution

Feel free to fork and submit pull requests. Ideas to add:
- SHAP value explanations
- Model export & prediction API
- Improved UI layout for Streamlit

---

## 📬 Contact

Made with ❤️ by [@siddsahu17](https://github.com/siddsahu17)  
Open to feedback, suggestions, or collaboration.

---

## 📄 License

This project is under the [MIT License](LICENSE).
