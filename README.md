# Prédiction du churn client

Ce projet consiste à prédire si un client d’un opérateur télécom va résilier son abonnement (churn).  
Il est conçu comme un projet ML complet pour montrer des compétences en prétraitement, modélisation, évaluation et déploiement.

---

## Structure du projet

ml-churn-prediction/
│
├── data/ # Dataset
│ └── WA*Fn-UseC*-Telco-Customer-Churn.csv
│
├── notebooks/ # Notebooks pour exploration et entraînement
│ ├── 01_exploration.ipynb # Analyse exploratoire et nettoyage
│ └── 02_model_training.ipynb # Prétraitement, entraînement, évaluation
│
├── src/ # Fonctions utilitaires
│ ├── preprocess.py # Fonctions de prétraitement
│ ├── train_model.py # Fonctions pour entraîner et sauvegarder les modèles
│ └── evaluate.py # Fonctions pour évaluer les modèles
│
├── model/ # Objets sauvegardés pour déploiement
│ ├── churn_model.pkl # Modèle Random Forest entraîné
│ ├── churn_xgb.pkl # Modèle XGBoost entraîné (optionnel)
│ ├── feature_columns.pkl # Colonnes/features utilisées à l'entraînement
│ ├── scaler.pkl # Scaler pour les colonnes numériques
│ ├── le_gender.pkl # Encoders pour toutes les colonnes catégorielles
│ ├── le_Partner.pkl
│ ├── le_Dependents.pkl
│ ├── le_PhoneService.pkl
│ ├── le_MultipleLines.pkl
│ ├── le_InternetService.pkl
│ ├── le_OnlineSecurity.pkl
│ ├── le_OnlineBackup.pkl
│ ├── le_DeviceProtection.pkl
│ ├── le_TechSupport.pkl
│ ├── le_StreamingTV.pkl
│ ├── le_StreamingMovies.pkl
│ ├── le_Contract.pkl
│ ├── le_PaperlessBilling.pkl
│ └── le_PaymentMethod.pkl
│
├── app/ # Application Streamlit
│ └── app.py # Interface pour prédire le churn
│
├── requirements.txt # Dépendances Python
└── README.md # Présentation complète du projet

---

## Dataset

- Source : [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contient des informations sur les clients, leurs abonnements, facturation et résiliation.
- La variable cible : `Churn` (Yes / No)

---

## Installation

1. Cloner le repository :

git clone https://github.com/jospindev-stack/ml-churn-prediction.git

cd ml-churn-prediction
Créer un environnement virtuel :
python -m venv venv

# Windows

venv\Scripts\activate

# Mac/Linux

source venv/bin/activate
Installer les dépendances :

pip install -r requirements.txt

Utilisation

Notebooks
01_exploration.ipynb : Analyse exploratoire des données et nettoyage

02_model_training.ipynb : Prétraitement, entraînement de Random Forest et XGBoost, évaluation et sauvegarde du modèle

Application Streamlit
Pour tester le modèle sur de nouvelles données :

streamlit run app/app.py
Remplir les champs du formulaire

Obtenir la prédiction et la probabilité que le client résilie son abonnement

Fonctionnalités
Prétraitement automatique des données

Encodage des variables catégorielles et standardisation des variables numériques

Entraînement et comparaison de deux modèles : Random Forest et XGBoost

Évaluation complète avec accuracy, classification report et matrice de confusion

Déploiement via Streamlit pour interface utilisateur
