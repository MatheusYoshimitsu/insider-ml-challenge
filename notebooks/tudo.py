Full pipeline for EDA and pre processing:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

%matplotlib inline

train_df = pd.read_csv("../dataset/train.csv")
test_df = pd.read_csv("../dataset/test.csv")

train_df["number_of_cabins"] = train_df.Cabin.apply(
    lambda x: 0 if pd.isna(x) else len(x.split(" "))
)
train_df["cabin_categories"] = train_df.Cabin.apply(lambda x: str(x)[0])
# Extract title from Name
train_df["name_title"] = train_df["Name"].apply(
    lambda x: x.split(",")[1].split(".")[0].strip()
)
train_df["train_test_join"] = 1
test_df["train_test_join"] = 0

test_df["Survived"] = -1  # Creating column Survived in test split
train_test_df = pd.concat(
    [train_df, test_df]
)  # We will group them for general analysis
train_test_df["Embarked"] = train_test_df["Embarked"].fillna("C")
# Fill null with the median age and fare
train_test_df.Age = train_test_df.Age.fillna(train_test_df.Age.median())
train_test_df.Fare = train_test_df.Fare.fillna(train_test_df.Fare.median())

# Normalizing Fare
train_test_df["norm_fare"] = np.log(train_test_df.Fare + 1)
# Count how many cabins a person has (0 if missing, otherwise count how many space-separated entries)
train_test_df["cabin_multiple"] = train_test_df.Cabin.apply(
    lambda x: 0 if pd.isna(x) else len(x.split(" "))
)

# Take the first letter of the cabin string (e.g., 'C85' → 'C'; NaN becomes 'n')
train_test_df["cabin_categories"] = train_test_df.Cabin.apply(
    lambda x: str(x)[0]
)

train_test_df.drop(columns=["Ticket"], inplace=True, errors='ignore') # If Ticket does is already dropped, ignores error

# Extract the title from the name (e.g., "Mr", "Mrs", "Miss", etc.)
train_test_df["name_title"] = train_test_df.Name.apply(
    lambda x: x.split(",")[1].split(".")[0].strip()
)
# Converting
train_test_df["cabin_multiple"] = train_test_df["cabin_multiple"].astype(str)
train_test_df.Pclass = train_test_df.Pclass.astype(str)

# Define categorical and numeric features
categorical_cols = [
    "Pclass",
    "Sex",
    "Embarked",
    "cabin_categories",
    "cabin_multiple",
    "name_title",
]
numerical_cols = [
    "Age",
    "SibSp",
    "Parch",
    "norm_fare",
]
meta_col = ["train_test_join"]  # used to split train/test later

# Apply one-hot encoding only to categorical columns
# We already done it to good categories for analysis purposes, but we did not merged them with our train before. This time, we are going to one hot encode all of them.
categorical_dummies = pd.get_dummies(train_test_df[categorical_cols])

# Combine with numerical + meta columns
train_test_dummies = pd.concat(
    [
        train_test_df[numerical_cols + meta_col].reset_index(drop=True),
        categorical_dummies.reset_index(drop=True),
    ],
    axis=1,
)
# Split train and test
X_full_train = train_test_dummies[train_test_dummies.train_test_join == 1].drop(
    ["train_test_join"], axis=1
)
X_test = train_test_dummies[train_test_dummies.train_test_join == 0].drop(
    ["train_test_join"], axis=1
)
y_full_train = train_test_df[train_test_df.train_test_join == 1].Survived

# Split training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train,
    y_full_train,
    test_size=0.2,
    random_state=29,
    stratify=y_full_train,
)

# Scale data
scale = StandardScaler()

# Columns to scale
columns_to_scale = ["Age", "SibSp", "Parch", "norm_fare"]

# Fit scaler only on X_train, then transform all
X_train[columns_to_scale] = scale.fit_transform(X_train[columns_to_scale])
X_val[columns_to_scale] = scale.transform(X_val[columns_to_scale])
X_test[columns_to_scale] = scale.transform(X_test[columns_to_scale])

rf_model = RandomForestClassifier(random_state=29)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)

# Evaluation
print("Random Forest Evaluation:")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print("F1 Score:", f1_score(y_val, y_pred_rf))
print("Precision:", precision_score(y_val, y_pred_rf))
print("Recall:", recall_score(y_val, y_pred_rf))

svm_model = SVC(probability = True, random_state=29)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_val)

# Evaluation
print("SVC Evaluation:")
print("Accuracy:", accuracy_score(y_val, y_pred_svm))
print("F1 Score:", f1_score(y_val, y_pred_svm))
print("Precision:", precision_score(y_val, y_pred_svm))
print("Recall:", recall_score(y_val, y_pred_svm))

def clf_performance(model, name):
    print(f"\nBest {name} Model Evaluation:")
    print("Best parameters:", model.best_params_)

    y_pred = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred))
    print("Recall:", recall_score(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))

rf = RandomForestClassifier(random_state=29)
param_grid_rf = {
    "n_estimators": [100, 200, 400, 500],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True],
    "max_depth": [15, 20, 25],
    "max_features": ["log2", "sqrt", 10],
    "min_samples_leaf": [2, 3],
    "min_samples_split": [2, 3],
}

clf_rf = GridSearchCV(rf, param_grid=param_grid_rf, cv=5, verbose=2, n_jobs=-1)
best_clf_rf = clf_rf.fit(X_train, y_train)
clf_performance(best_clf_rf, "Random Forest")

# SVC
svc = SVC(probability=True, random_state=29)
param_grid_svc = [
    {
        "kernel": ["rbf"],
        "gamma": [0.1, 0.5, 1, 2, 5, 10],
        "C": [0.1, 1, 10, 100, 1000],
    },
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [2, 3, 4, 5], "C": [0.1, 1, 10, 100, 1000]},
]

clf_svc = GridSearchCV(
    svc, param_grid=param_grid_svc, cv=5, verbose=2, n_jobs=-1
)
best_clf_svc = clf_svc.fit(X_train, y_train)
clf_performance(best_clf_svc, "SVC")

joblib.dump(best_clf_svc.best_estimator_, "../models/best_svc_model.pkl")
joblib.dump(best_clf_rf.best_estimator_, "../models/best_rf_model.pkl")