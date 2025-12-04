# Credit Card Fraud Detection Model

This project focuses on detecting fraudulent credit card transactions using machine learning on a highly imbalanced dataset. The dataset includes both legitimate and fraudulent transactions, with fraudulent transactions being significantly less frequent.

## Project Description

This project aims to detect fraudulent credit card transactions using multiple machine learning techniques on a highly imbalanced dataset. The dataset contains a mix of legitimate and fraudulent transactions, where fraudulent transactions represent only 0.17% of all transactions. The project compares various sampling strategies and algorithms to achieve optimal fraud detection performance.

## Key Steps

### 1) Data Loading and Exploration
- Load the dataset (`creditcard.csv`) into a Pandas DataFrame
- Inspect the dataset's structure and check for missing values
- Analyze the distribution of legitimate (0) and fraudulent (1) transactions
- Perform statistical analysis on transaction amounts for both classes

### 2) Data Preprocessing
- Handle missing values by removing incomplete records
- Separate the dataset into legitimate and fraudulent transaction DataFrames
- Address class imbalance through two approaches:
  - **Random Undersampling**: Reduces majority class to match minority class
  - **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates synthetic fraud samples

### 3) Feature and Target Separation
- Define features (X) by excluding the `Class` column
- Define the target (Y) as the `Class` column
- Features include: Time, Amount, and V1-V28 (PCA-transformed anonymized features)

### 4) Data Splitting
- Split the data into training and testing sets (80-20 split)
- Use stratification to maintain class distribution in both sets

### 5) Model Training
Train multiple machine learning models:
- **Logistic Regression**: Simple, interpretable baseline model
- **Random Forest**: Ensemble method for capturing complex patterns

Each model is trained with both undersampling and SMOTE techniques for comparison.

### 6) Model Evaluation
Evaluate models using comprehensive metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted frauds that are actual frauds
- **Recall**: Proportion of actual frauds that are detected (most critical)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to discriminate between classes
- **Confusion Matrix**: Detailed breakdown of predictions

## Results

The project demonstrates highly effective fraud detection across multiple approaches:

**Key Findings:**
- Random Forest with SMOTE provides the best overall performance
- High recall rates across all models ensure fraudulent transactions are caught
- Top important features identified: V14, V17, V12, V10, V11

## Dataset

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Dependencies

The project requires the following Python libraries:

```bash
numpy
pandas
scikit-learn
imbalanced-learn
```

Install the dependencies using:

```bash
pip install numpy pandas scikit-learn imbalanced-learn
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/sakshammgarg/CreditCard_FraudDetection.git
cd CreditCard_FraudDetection
```

2. **Download the dataset**
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in the project root directory

## Usage

Run the notebook to explore the complete analysis.

The notebook will:
1. Load and explore the credit card transaction dataset
2. Preprocess data and handle class imbalance using undersampling and SMOTE
3. Train Logistic Regression and Random Forest models
4. Display comprehensive evaluation metrics for all model combinations
5. Show feature importance analysis
6. Provide model comparison summary and recommendations

## Project Structure

```
CreditCard_FraudDetection/
│
├── CreditCard_FraudDetection.ipynb   # Main notebook
├── creditcard.csv                    # Dataset (download separately)
└── README.md                         # Project documentation
```

## Model Selection Guide

**For Production Deployment:**
- **Maximum Fraud Detection**: Random Forest with SMOTE (highest recall)
- **Balanced Performance**: Random Forest with SMOTE (best F1-score)
- **Fast Predictions**: Logistic Regression with undersampling
- **Minimal False Alarms**: Random Forest with undersampling (highest precision)

## Advanced Features

This enhanced version includes:
- Multiple sampling techniques comparison
- Four different model-sampling combinations
- Comprehensive evaluation metrics beyond accuracy
- Feature importance analysis
- Confusion matrix with detailed interpretation
- Model comparison summary
- Production deployment recommendations

## Future Improvements

Potential enhancements for even better results:
1. Implement XGBoost or LightGBM models
2. Hyperparameter tuning using GridSearchCV
3. Ensemble methods combining multiple models
4. Cost-sensitive learning with weighted penalties
5. Deep learning approaches (Neural Networks)
6. Real-time fraud detection API
7. Model monitoring and drift detection

---

This project provides a robust and comprehensive approach to credit card fraud detection using machine learning, highlighting the importance of addressing class imbalance, using multiple evaluation metrics, and comparing different modeling approaches for optimal performance.
