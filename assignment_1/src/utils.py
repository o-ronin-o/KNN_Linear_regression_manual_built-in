import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
from src.knn_manual import KNN_predict
def load_dataset(path):
    return pd.read_csv(path)

def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=seed)
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_ratio, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred):
   
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def evaluate_knn_for_ks(X_train, y_train, X_val, y_val, k_values):
  
    results = []

    for k in k_values:
        y_pred_val = KNN_predict(X_train, y_train, X_val, k)
        metrics = classification_metrics(y_val, y_pred_val)
        metrics["k"] = k
        results.append(metrics)

    return results


def regression_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }
