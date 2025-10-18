# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path="../data/clean_telco.csv"):
    """Charge le dataset nettoyé"""
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Prétraitement des features :
    - Encodage des variables catégorielles
    - Standardisation des variables numériques
    """
    # Séparer les features et la cible
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identifier les colonnes numériques et catégorielles
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Encodage des variables catégorielles
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Standardisation des variables numériques
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Séparation train/test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test