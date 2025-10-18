# src/train_model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_random_forest(X_train, y_train, save_model_path="../model/churn_rf.pkl", save_features_path="../model/feature_columns.pkl"):
    """
    Entraîne un Random Forest et sauvegarde le modèle et les colonnes utilisées.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Sauvegarde du modèle
    joblib.dump(rf, save_model_path)
    # Sauvegarde des colonnes
    joblib.dump(X_train.columns.tolist(), save_features_path)
    
    print(f"Random Forest sauvegardé dans {save_model_path}")
    return rf

def train_xgboost(X_train, y_train, save_model_path="../model/churn_xgb.pkl"):
    """
    Entraîne un XGBoost et sauvegarde le modèle.
    """
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_clf.fit(X_train, y_train)
    
    joblib.dump(xgb_clf, save_model_path)
    print(f"XGBoost sauvegardé dans {save_model_path}")
    return xgb_clf
