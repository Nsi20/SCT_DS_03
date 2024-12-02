# SCT_DS_03
Analysing the Bank Marketing dataset to build a predictive model for customer term deposit subscription." (Focuses on dataset and objective)

### Predicting Customer Purchase Intentions Using Logistic Regression and SMOTE

---

#### **Project Overview**

This project aims to predict whether a customer will purchase a product or service based on their demographic and behavioural data. Using machine learning techniques, we analyze the data to identify patterns and build a robust predictive model. The focus is on handling class imbalance effectively using **SMOTE (Synthetic Minority Oversampling Technique)** and evaluating the performance of various models, starting with Logistic Regression and Random Forest Classifiers.

---

#### **Key Features**

- **Dataset**: The dataset contains customer demographic and behavioural attributes such as age, job, marital status, education, balance, and more. It also includes the target variable `y` (whether the customer made a purchase).
- **Data Preprocessing**: Handling missing values, encoding categorical variables using one-hot encoding, and scaling numerical data for better model performance.
- **Imbalanced Data Handling**: The dataset is imbalanced, with significantly fewer positive (purchase) cases. SMOTE is used to synthetically generate samples of the minority class to improve prediction performance.
- **Model Training**: Logistic Regression is initially used as the baseline model, followed by Random Forest for comparison.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are used to evaluate the models.

---

#### **Key Insights**

1. The dataset is imbalanced, with the majority class (`no purchase`) dominating.
2. Logistic Regression struggles to predict the minority class effectively without SMOTE.
3. Applying SMOTE improved recall for the minority class significantly at the cost of slightly reduced overall accuracy.
4. Random Forest shows potential for better performance in terms of classifying minority cases.

---

#### **Project Workflow**

1. **Data Loading**: Fetching and exploring the dataset.
2. **Data Cleaning**:
   - Handling missing values.
   - Encoding categorical variables using one-hot encoding.
   - Standardizing numerical features.
3. **Exploratory Data Analysis**:
   - Distribution of features.
   - Imbalance in the target variable.
   - Correlation between features.
4. **Model Development**:
   - Logistic Regression as a baseline model.
   - Addressing imbalance using SMOTE.
   - Evaluating Logistic Regression with and without SMOTE.
   - Trying alternative models like Random Forest for improved performance.
5. **Performance Evaluation**:
   - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. **Conclusion and Next Steps**:
   - Identifying areas for improvement (e.g., feature engineering, trying other models like XGBoost).

---

#### **Results**

- **Logistic Regression without SMOTE**:
  - Accuracy: 89.8%
  - Recall (Class 1): 27%
  - F1-Score (Class 1): 36%

- **Logistic Regression with SMOTE**:
  - Accuracy: 83.3%
  - Recall (Class 1): 78%
  - F1-Score (Class 1): 50%

- **Random Forest with SMOTE** *(proposed next step)*:
  - Expected to balance performance across both classes better.

---

#### **Technologies and Libraries**

- **Languages**: Python
- **Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `matplotlib` and `seaborn` for data visualization.
  - `scikit-learn` for machine learning models and performance evaluation.
  - `imbalanced-learn` for SMOTE implementation.

---

#### **Folder Structure**

```
├── data/
│   └── bank_marketing.csv    # Dataset file
├── notebooks/
│   └── customer_purchase_prediction.ipynb  # Main Jupyter Notebook
├── src/
│   ├── preprocessing.py      # Data cleaning and preprocessing scripts
│   ├── modeling.py           # Training and evaluation scripts
│   └── utils.py              # Helper functions
├── README.md                 # Project documentation
└── requirements.txt          # Required libraries
```

---

#### **How to Run**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/customer-purchase-prediction.git
   cd customer-purchase-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   Open `customer_purchase_prediction.ipynb` in Jupyter Notebook or Google Colab and follow the steps to preprocess the data, train models, and evaluate performance.

---

#### **Future Work**

1. Experiment with advanced models like **XGBoost** or **LightGBM**.
2. Implement hyperparameter tuning for Random Forest and Logistic Regression.
3. Introduce additional features (e.g., interaction terms, derived variables) to improve prediction accuracy.
4. Deploy the best-performing model using **Flask** or **FastAPI** for real-time predictions.

---

#### **Contributors**

- **Nsidibe Daniel Essang**  
  *Data Analyst | Business Analyst | Expert in Python, SQL, Power BI, and Machine Learning*

---
