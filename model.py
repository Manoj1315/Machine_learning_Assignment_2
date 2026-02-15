import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
data = pd.read_csv("/home/cloud/Downloads/ML Assignent/Data/ClinicalTrial.csv") 

target_column = "trt"   

if target_column not in data.columns:
    raise ValueError(f"'{target_column}' not found in dataset columns!")

data = data.dropna(subset=[target_column])

X = data.drop(target_column, axis=1)
y = data[target_column]

# --------------------------------------------------
# 2. OPTIMIZATION: HANDLE MISSING FEATURES
# --------------------------------------------------
# Clinical datasets almost always have missing values. We fill them before encoding.
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Fill missing numeric values with the median, and categorical with the mode
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# --------------------------------------------------
# 3. OPTIMIZATION: ONE-HOT ENCODING
# --------------------------------------------------
# Replaces LabelEncoder to prevent the models from making false math assumptions.
# drop_first=True prevents "dummy variable trap" (multicollinearity).
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

# Encode the Target into distinct 0, 1, 2... classes
y = LabelEncoder().fit_transform(y.astype(str))
y = pd.Series(y)

# Remove classes with less than 2 samples (required for stratify)
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
valid_indices = y.isin(valid_classes)

X = X[valid_indices]
y = y[valid_indices]

num_classes = len(np.unique(y))
is_multiclass = num_classes > 2
avg_method = "weighted" if is_multiclass else "binary"

print(f"Target has {num_classes} classes. Using average='{avg_method}' for metrics.")

# --------------------------------------------------
# 4. TRAIN TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 5. OPTIMIZATION: SMART SCALING
# --------------------------------------------------
# We ONLY scale the originally continuous/numeric columns. 
# We do not scale the One-Hot encoded 0s and 1s, preserving their structure.
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if len(numeric_cols) > 0:
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --------------------------------------------------
# 6. OPTIMIZATION: MODEL TUNING & CLASS BALANCING
# --------------------------------------------------
xgb_eval_metric = "mlogloss" if is_multiclass else "logloss"

models = {
    # class_weight='balanced' forces models to pay attention to rare classes
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    
    # max_depth limits prevent the tree from overfitting the training data
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
    
    # weights='distance' gives closer neighbors more voting power
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    
    "Naive Bayes": GaussianNB(),
    
    # Increased estimators and limited depth for a more robust forest
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42),
    
    # Added explicit learning rate and max_depth to control complexity
    "XGBoost": XGBClassifier(eval_metric=xgb_eval_metric, learning_rate=0.1, max_depth=6, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    auc = "NA"
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test_scaled)
            if is_multiclass:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average=avg_method)
            else:
                auc = roc_auc_score(y_test, y_prob[:, 1])
        except Exception as e:
            auc = "NA"

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=avg_method, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg_method, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg_method, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")

joblib.dump(scaler, "scaler.pkl")

print("\nModels trained and saved successfully!")
print(pd.DataFrame(results).T)