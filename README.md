# Machine_learning_Assignment_2
2024dc04079 Machine_learning_Assignment_2, Clinicaltrials classification model streamlit app


 AIDS Clinical Trial - Machine Learning Classification Project

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Model Performance Comparison](#model-performance-comparison)
- [Observations and Analysis](#observations-and-analysis)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Conclusion](#conclusion)

---

## Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict treatment outcomes in HIV-infected patients based on clinical trial data. Specifically, the goal is to classify patient infection status using various patient health metrics, treatment information, and demographic data collected from the AIDS Clinical Trials Group study.

The study focuses on comparing the efficacy of different AIDS treatments (AZT, ddI, and ddC) in preventing disease progression among HIV-infected patients with CD4 cell counts ranging from 200-500 cells/mm³. By applying multiple machine learning algorithms, this project aims to:

1. Identify patterns and relationships in clinical trial data that predict patient outcomes
2. Compare the performance of six different machine learning models
3. Determine which model provides the most reliable predictions for clinical decision-making
4. Provide insights into feature importance for patient prognosis

This classification problem is critical for healthcare decision-making, as accurate predictions can help clinicians personalize treatment strategies and improve patient outcomes.

---

## Dataset Description

### Source
**Dataset Name:** AIDS Clinical Trials Dataset  
**Source:** [Kaggle - AIDS Clinical Trials](https://www.kaggle.com/datasets/tanshihjen/aids-clinical-trials?resource=download)  
**Study Type:** Interventional (Clinical Trial)  
**Study Design:** Randomized, Double-Blind Phase II/III Trial

### Dataset Overview
- **Number of Instances:** 2,139 patient records
- **Number of Features:** 24 attributes
- **Target Variable:** `infected` (4 classes - multi-class classification)
- **Study Period:** Completed November 1995
- **Enrollment:** 2,100 participants
- **Masking:** Double-Blind
- **Primary Purpose:** Treatment evaluation

### Study Objectives
The dataset was created to evaluate the efficacy and safety of various AIDS treatments, specifically comparing:
- **AZT (Zidovudine)** monotherapy
- **ddI (Didanosine)** monotherapy
- **ddC (Zalcitabine)** in combination with AZT
- **Combination therapy** with nucleoside analogs

The goal was to prevent disease progression in HIV-infected patients with CD4 counts between 200-500 cells/mm³.

### Funding Sources
- AIDS Clinical Trials Group of the National Institute of Allergy and Infectious Diseases
- General Research Center units funded by the National Center for Research Resources

### Key Attributes

#### 1. **Patient Status Information**
- **Censoring Indicator (label):** Binary indicator (1 = failure, 0 = censoring)

#### 2. **Temporal Information**
- **Time to Event (time):** Integer representing time to failure or censoring

#### 3. **Treatment Features**
- **Treatment Indicator (trt):** Categorical feature
  - 0 = ZDV only
  - 1 = ZDV + ddI
  - 2 = ZDV + Zalcitabine
  - 3 = ddI only

#### 4. **Baseline Health Metrics**
- **Age (age):** Patient's age in years at baseline
- **Weight (wtkg):** Weight in kilograms at baseline
- **Hemophilia (hemo):** Binary indicator (0 = no, 1 = yes)
- **Sexual Orientation (homo):** Binary indicator of homosexual activity (0 = no, 1 = yes)
- **IV Drug Use History (drugs):** Binary indicator (0 = no, 1 = yes)
- **Karnofsky Score (karnof):** Integer performance status scale
- **CD4 Count:** Baseline immune system measurement
- **Additional clinical and demographic variables**

### Data Preprocessing
- **Categorical Encoding:** Label encoding applied to categorical features
- **Feature Scaling:** StandardScaler used for normalization
- **Train-Test Split:** 80-20 split with stratification
- **Class Imbalance Handling:** Removed classes with fewer than 2 samples

---

## Models Used

Six machine learning algorithms were implemented and evaluated for this multi-class classification problem:

### 1. **Logistic Regression**
- Linear model for classification
- Maximum iterations: 1000
- Suitable for baseline comparison

### 2. **Decision Tree Classifier**
- Non-parametric model using tree-based decisions
- Captures non-linear relationships
- Interpretable feature importance

### 3. **K-Nearest Neighbors (kNN)**
- Instance-based learning algorithm
- Classification based on proximity to training samples
- Non-parametric approach

### 4. **Naive Bayes (Gaussian)**
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction

### 5. **Random Forest (Ensemble)**
- Ensemble of decision trees
- Reduces overfitting through bagging
- Robust to outliers and noise

### 6. **XGBoost (Ensemble)**
- Gradient boosting framework
- Advanced ensemble technique
- Handles complex patterns and interactions

All models were trained using weighted averaging for multi-class metrics due to the 4-class target variable.

---

## Model Performance Comparison

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|---------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.5210 | 0.7525 | 0.5218 | 0.5210 | 0.5205 | 0.3623 |
| **Decision Tree** | 0.5187 | 0.7180 | 0.5170 | 0.5187 | 0.5090 | 0.3653 |
| **kNN** | 0.5070 | 0.7046 | 0.5129 | 0.5070 | 0.5096 | 0.3424 |
| **Naive Bayes** | 0.4743 | 0.7483 | 0.4740 | 0.4743 | 0.4718 | 0.3010 |
| **Random Forest (Ensemble)** | 0.4743 | 0.7353 | 0.4765 | 0.4743 | 0.4748 | 0.2988 |
| **XGBoost (Ensemble)** | 0.4930 | 0.7531 | 0.4927 | 0.4930 | 0.4921 | 0.3239 |

### Metric Definitions

- **Accuracy:** Overall correctness of predictions (TP + TN) / Total
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes (0.5-1.0)
- **Precision:** Proportion of positive predictions that are correct (TP / (TP + FP))
- **Recall:** Proportion of actual positives correctly identified (TP / (TP + FN))
- **F1 Score:** Harmonic mean of precision and recall (2 × (Precision × Recall) / (Precision + Recall))
- **MCC (Matthews Correlation Coefficient):** Balanced measure considering all confusion matrix elements (-1 to +1)

---

## Observations and Analysis

### Model Performance Summary

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | **Best overall performer** with the highest accuracy (52.10%) and competitive AUC (0.7525). Despite being a simple linear model, it achieves balanced performance across all metrics with the best F1 score (0.5205) and strong MCC (0.3623). This suggests that linear boundaries can reasonably separate the classes in this dataset. The model provides good interpretability and computational efficiency, making it suitable for clinical deployment. |
| **Decision Tree** | Achieves the **highest MCC (0.3653)**, indicating the best balance across true/false positives and negatives. With accuracy of 51.87% and moderate AUC (0.7180), it captures non-linear patterns effectively. However, the F1 score (0.5090) is lower than Logistic Regression, suggesting some trade-off between precision and recall. Decision trees offer excellent interpretability through visualization of decision rules, which is valuable for clinical understanding. |
| **kNN** | Shows **moderate performance** with 50.70% accuracy and the lowest AUC (0.7046) among all models. The instance-based learning approach struggles with this high-dimensional clinical data, possibly due to the curse of dimensionality. The F1 score (0.5096) and MCC (0.3424) indicate reasonable but not optimal performance. kNN may require feature selection or dimensionality reduction to improve performance on this dataset. |
| **Naive Bayes** | Despite having the **lowest accuracy (47.43%)**, it achieves the **second-best AUC (0.7483)**, indicating good probabilistic ranking of predictions. This paradox suggests that while the hard classifications are often incorrect, the probability estimates are well-calibrated. The assumption of feature independence likely doesn't hold well for correlated clinical variables, limiting its classification accuracy. However, its strong AUC makes it useful for risk stratification rather than definitive classification. |
| **Random Forest (Ensemble)** | **Surprisingly underperforms** with the lowest accuracy (47.43%) and lowest MCC (0.2988) among all models, despite being an ensemble method. This unexpected result may indicate overfitting to noise in the training data or that the bagging approach doesn't capture the underlying patterns effectively. With moderate AUC (0.7353), it shows some discriminative ability but fails to translate this into accurate predictions. Hyperparameter tuning (tree depth, number of estimators) may be needed to improve performance. |
| **XGBoost (Ensemble)** | Shows **mixed performance** with the **highest AUC (0.7531)** but moderate accuracy (49.30%). This indicates excellent ability to rank predictions by probability, making it valuable for identifying high-risk patients. However, the accuracy and F1 score (0.4921) suggest suboptimal hard classification thresholds. The MCC (0.3239) is mid-range, indicating balanced but not exceptional performance. XGBoost may benefit from threshold optimization and hyperparameter tuning to convert its strong discriminative ability into better classification accuracy. |

### Overall Insights

1. **Model Selection:** Logistic Regression emerges as the best choice for this dataset, balancing accuracy, interpretability, and computational efficiency.

2. **AUC vs. Accuracy Trade-off:** Several models (XGBoost, Naive Bayes) show high AUC but lower accuracy, suggesting they're better at probability estimation than hard classification. This is valuable for clinical risk assessment.

3. **Ensemble Performance:** Surprisingly, ensemble methods (Random Forest, XGBoost) don't significantly outperform simpler models, suggesting:
   - The dataset may be relatively simple with linear separability
   - Ensemble complexity may be introducing overfitting
   - Hyperparameter optimization is needed

4. **Dataset Challenges:** 
   - Moderate accuracy across all models (47-52%) indicates inherent classification difficulty
   - The 4-class target may have overlapping feature distributions
   - Clinical data variability and noise may limit predictive power

5. **Clinical Implications:** 
   - Models with high AUC (XGBoost: 0.7531, Logistic Regression: 0.7525) are suitable for risk stratification
   - Logistic Regression provides the best balance for treatment decision support
   - No model achieves excellent performance, highlighting the complexity of predicting AIDS treatment outcomes

6. **Recommendations:**
   - Deploy Logistic Regression for production use due to best overall performance
   - Use XGBoost/Naive Bayes for probability-based risk assessment
   - Further improve models through feature engineering, hyperparameter tuning, and class balancing techniques
   - Consider combining multiple models in a voting ensemble

---

## Project Structure

```
ML Assignment/
│
├── model.py                    # Main training script
├── README.md                   # This file
├── ClinicalTrial.csv          # Dataset (not included in repo)
│
├── Models/                     # Saved model files
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── KNN.pkl
│   ├── Naive_Bayes.pkl
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   └── scaler.pkl             # StandardScaler object
│
└── requirements.txt           # Python dependencies
```

---

### Output
```
Target has 4 classes. Using average='weighted' for metrics.
Training Logistic Regression...
Training Decision Tree...
Training KNN...
Training Naive Bayes...
Training Random Forest...
Training XGBoost...

Models trained and saved successfully!

                     Accuracy       AUC  Precision    Recall        F1       MCC
Logistic Regression  0.521028  0.752467   0.521769  0.521028  0.520523  0.362275
Decision Tree        0.518692  0.717962   0.516962  0.518692  0.508982  0.365305
KNN                  0.507009  0.704615   0.512921  0.507009  0.509611  0.342364
Naive Bayes          0.474299  0.748337   0.473975  0.474299  0.471768  0.301048
Random Forest        0.474299  0.735279   0.476450  0.474299  0.474787  0.298762
XGBoost              0.492991  0.753099   0.492681  0.492991  0.492123  0.323899
```

---

## Conclusion

This project successfully implements and compares six machine learning models for predicting AIDS treatment outcomes based on clinical trial data. Key findings include:

1. **Logistic Regression** achieves the best overall performance (52.10% accuracy, 0.7525 AUC)
2. **XGBoost** provides the best probability estimates (0.7531 AUC) for risk assessment
3. **Simple models outperform complex ensembles**, suggesting linear separability in the feature space
4. **Moderate performance across all models** (47-52% accuracy) indicates the inherent complexity of clinical outcome prediction

The models demonstrate practical applicability for:
- Clinical decision support systems
- Patient risk stratification
- Treatment outcome prediction
- Research into AIDS treatment effectiveness

---

**Project Author:** Manoj M  
**Date:** February 15, 2026  
**Course:** Machine Learning Assignment  
**Dataset Source:** [Kaggle - AIDS Clinical Trials](https://www.kaggle.com/datasets/tanshihjen/aids-clinical-trials)
Kaggle Dataset: https://www.kaggle.com/datasets/tanshihjen/aids-clinical-trials
Streamlit app link: https://machinelearningassignment2-mogutlcen7azbxtkphtjjx.streamlit.app/

4. Scikit-learn Documentation: https://scikit-learn.org/

5. XGBoost Documentation: https://xgboost.readthedocs.io/
