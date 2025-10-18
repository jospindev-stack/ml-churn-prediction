# Prédiction du churn client

Ce projet consiste à prédire si un client d’un opérateur télécom va résilier son abonnement (churn).  
Il est conçu comme un projet ML complet pour montrer des compétences en prétraitement, modélisation, évaluation et déploiement.

---

## Structure du projet

```
ml-churn-prediction/
├── data/                         # Données brutes et nettoyées
│   └── WA*Fn-UseC*-Telco-Customer-Churn.csv
│
├── notebooks/                   # Notebooks d'analyse et d'entraînement
│   ├── 01_exploration.ipynb         # Analyse exploratoire, visualisations
│   └── 02_model_training.ipynb      # Prétraitement, entraînement, évaluation
│
├── src/                         # Fonctions Python modulaires
│   ├── preprocess.py                # Chargement, nettoyage, encodage
│   ├── train_model.py               # Entraînement et sauvegarde des modèles
│   └── evaluate.py                  # Métriques et visualisations d'évaluation
│
├── model/                       # Objets sauvegardés pour déploiement
│   ├── churn_model.pkl              # Modèle Random Forest
│   ├── churn_xgb.pkl                # Modèle XGBoost (optionnel)
│   ├── churn_pipeline.pkl          # Pipeline complet (prétraitement + modèle)
│   ├── feature_columns.pkl          # Liste des colonnes utilisées
│   ├── scaler.pkl                   # Scaler pour les variables numériques
│   ├── le_<col>.pkl                 # Encoders LabelEncoder pour chaque variable catégorielle
│
├── app/                         # Application Streamlit
│   └── app.py                      # Interface utilisateur pour prédiction
│
├── requirements.txt             # Dépendances Python
└── README.md                    # Présentation du projet

```

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

# Utilisation

Notebooks
01_exploration.ipynb : Analyse exploratoire des données et nettoyage

02_model_training.ipynb : Prétraitement, entraînement de Random Forest et XGBoost, évaluation et sauvegarde du modèle

Application Streamlit
Pour tester le modèle sur de nouvelles données :

streamlit run app/app.py
Remplir les champs du formulaire

Obtenir la prédiction et la probabilité que le client résilie son abonnement

# Fonctionnalités

Prétraitement automatique des données

Encodage des variables catégorielles et standardisation des variables numériques

Entraînement et comparaison de deux modèles : Random Forest et XGBoost

Évaluation complète avec accuracy, classification report et matrice de confusion

Déploiement via Streamlit pour interface utilisateur
