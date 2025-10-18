# src/evaluate.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Évalue un modèle et affiche les métriques principales et la matrice de confusion.
    """
    y_pred = model.predict(X_test)
    
    print(f"---- {model_name} ----")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Classification Report :\n", classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité terrain")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()
    
    return y_pred
