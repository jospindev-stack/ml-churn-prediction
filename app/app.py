import streamlit as st
import pandas as pd
import joblib

# ----------------------------
#  Chargement du pipeline complet
# ----------------------------
pipeline = joblib.load("../model/churn_pipeline.pkl")

st.title(" Prédiction de churn client")

st.write("""
Cette application prédit si un client va résilier son abonnement.
Veuillez entrer les informations du client ci-dessous :
""")

# ----------------------------
#  Champs utilisateur
# ----------------------------
gender = st.selectbox("Genre", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partenaire", ["Yes", "No"])
dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
tenure = st.slider("Durée de service (mois)", 0, 72, 12)
phone_service = st.selectbox("Service téléphonique", ["Yes", "No"])
multiple_lines = st.selectbox("Lignes multiples", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Sauvegarde en ligne", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Protection des appareils", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Support technique", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming films", ["Yes", "No", "No internet service"])
contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Facturation sans papier", ["Yes", "No"])
payment_method = st.selectbox("Méthode de paiement", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Facturation mensuelle ($)", min_value=0.0, value=70.0)
total_charges = st.number_input("Facturation totale ($)", min_value=0.0, value=1000.0)

# ----------------------------
#  Création du DataFrame brut
# ----------------------------
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# ----------------------------
#  Prédiction via pipeline
# ----------------------------
prediction = pipeline.predict(input_df)[0]
prediction_prob = pipeline.predict_proba(input_df)[0][1]

# ----------------------------
#  Affichage du résultat
# ----------------------------
st.subheader("Résultat :")
if prediction == 1:
    st.error(f" Le client risque de churn (probabilité {prediction_prob:.2f})")
else:
    st.success(f" Le client est peu susceptible de churn (probabilité {prediction_prob:.2f})")
