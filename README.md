# Credit Card Fraud Detection Model

This project focuses on detecting fraudulent credit card transactions using machine learning on a **highly imbalanced dataset**, where fraudulent transactions account for only **0.17%** of all records. The project emphasizes correct handling of class imbalance, fraud-focused evaluation metrics, and comparative analysis of multiple modeling strategies.

---

## Project Description

This project aims to detect fraudulent credit card transactions using multiple machine learning techniques on a real-world, highly imbalanced dataset. It compares different **sampling strategies** and **models** to identify approaches that maximize fraud detection while maintaining reasonable false-positive rates.

---

## Key Steps

### 1) Data Loading and Exploration
- Load the dataset (`creditcard.csv`) into a Pandas DataFrame  
- Inspect dataset structure and check for missing values  
- Analyze class imbalance between legitimate (0) and fraudulent (1) transactions  
- Perform statistical analysis on transaction amounts  

---

### 2) Data Preprocessing
- Remove incomplete records (if any)  
- Separate legitimate and fraudulent transactions  
- Address class imbalance using:
  - **Random Undersampling**: Reduces majority class size  
  - **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic fraud samples  

---

### 3) Feature and Target Separation
- Features (X): All columns except `Class`  
- Target (Y): `Class`  
- Features include:
  - `Time`, `Amount`
  - `V1–V28` (PCA-transformed, anonymized features)

---

### 4) Data Splitting
- 80–20 train–test split  
- Stratified split to preserve class distribution  

---

### 5) Model Training
The following models are trained using both sampling strategies:

- **Logistic Regression**
  - Simple and interpretable baseline model  
- **Random Forest**
  - Ensemble method capable of capturing complex patterns  

This results in **four model–sampling combinations**:
- Logistic Regression + Undersampling  
- Logistic Regression + SMOTE  
- Random Forest + Undersampling  
- Random Forest + SMOTE  

---

### 6) Model Evaluation
Models are evaluated using **fraud-focused metrics**:

- **Accuracy** (reported for completeness)  
- **Precision** (false-alarm control)  
- **Recall** (most critical for fraud detection)  
- **F1-Score** (balanced performance metric)  
- **ROC-AUC**  
- **Confusion Matrix**  

> Decision thresholds are optimized to maximize F1-score, ensuring fair evaluation under severe class imbalance.

---

## Results

### Key Findings
- **Random Forest with SMOTE** achieves the highest recall and strongest fraud detection capability  
- **Random Forest with Undersampling** provides the best balance between precision and recall  
- Logistic Regression performs reasonably but is consistently outperformed by Random Forest  
- Most influential features across models:
  - **V14, V17, V12, V10, V11**

---

## Model Selection & Recommendations

### 1. Best Model Selection
- **Maximum Fraud Detection (Recall)**: Random Forest + SMOTE  
- **Balanced Performance (F1-score)**: Random Forest + Undersampling  
- **Fast & Interpretable Baseline**: Logistic Regression + Undersampling  
- **Production Benchmark**: Random Forest models  

---

### 2. Key Insights
- Random Forest consistently outperforms Logistic Regression  
- SMOTE improves recall but may increase false positives  
- Undersampling provides better precision stability  
- Threshold tuning significantly improves F1-score  
- PCA-based features (V14, V17, V12, V10) dominate fraud detection  

---

### 3. Production Considerations
- Monitor **data drift** and retrain periodically  
- Adjust decision thresholds based on business cost sensitivity  
- Perform **A/B testing** before deploying new models  
- Maintain explainability for compliance and audits  
- Implement alerting mechanisms for flagged transactions  

---

### 4. Next Steps / Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Advanced models (XGBoost, LightGBM)  
- Cost-sensitive learning with asymmetric penalties  
- Ensemble methods combining multiple models  
- Real-time inference pipeline  
- Model monitoring and automated retraining  

---

## Dataset

**Source:**  
[Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Dependencies

```bash
numpy
pandas
scikit-learn
imbalanced-learn

Install with:

pip install numpy pandas scikit-learn imbalanced-learn


⸻

Installation
	1.	Clone the repository

git clone https://github.com/sakshammgarg/CreditCard_FraudDetection.git
cd CreditCard_FraudDetection

	2.	Download the dataset

	•	Visit the Kaggle link above
	•	Download creditcard.csv
	•	Place it in the project root directory

⸻

Usage

Run the notebook to:
	1.	Load and explore the dataset
	2.	Handle class imbalance using undersampling and SMOTE
	3.	Train Logistic Regression and Random Forest models
	4.	Evaluate models using fraud-focused metrics
	5.	Analyze feature importance
	6.	Compare models and derive recommendations

⸻

Project Structure

CreditCard_FraudDetection/
│
├── CreditCard_FraudDetection.ipynb
├── creditcard.csv
└── README.md


⸻

Final Note

This project demonstrates that handling class imbalance correctly and using appropriate evaluation metrics is more important than maximizing raw accuracy. By combining sampling strategies, threshold optimization, and ensemble models, the system achieves robust and defensible fraud detection performance.

---
