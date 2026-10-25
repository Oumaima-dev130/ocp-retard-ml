
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

DOSSIER = Path(__file__).resolve().parent
SEUIL = 0.4

# Chargement du préprocesseur et des paramètres du modèle
preprocesseur = joblib.load(DOSSIER / "preprocesseur.pkl")
donnees_modele = np.load(DOSSIER / "modele_logistique_final.npz")

w = donnees_modele["w"]
b = float(donnees_modele["b"])

print("Préprocesseur chargé.")
print("Nombre de coefficients :", len(w))
print("Biais :", b)
print("Seuil :", SEUIL)


@app.route("/", methods=["GET"])
def accueil():
    return jsonify({
        "message": "API de prédiction des retards OCP opérationnelle"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        donnees = request.get_json(silent=True)

        if not isinstance(donnees, dict):
            return jsonify({
                "erreur": "Envoyez un objet JSON valide."
            }), 400

        colonnes = [
            "dureeV0",
            "ponderation",
            "effectif",
            "retard_predecesseur",
            "rubrique",
            "responsable"
        ]

        manquantes = [
            c for c in colonnes
            if c not in donnees or donnees[c] is None
            or donnees[c] == ""
        ]

        if manquantes:
            return jsonify({
                "erreur": "Certaines variables sont manquantes.",
                "variables_manquantes": manquantes
            }), 400

        # Vérifier les variables numériques
        try:
            valeurs_numeriques = {
                "dureeV0": float(donnees["dureeV0"]),
                "ponderation": float(donnees["ponderation"]),
                "effectif": float(donnees["effectif"]),
                "retard_predecesseur": float(
                    donnees["retard_predecesseur"]
                )
            }
        except (ValueError, TypeError):
            return jsonify({
                "erreur": "Les quatre variables numériques doivent être des nombres."
            }), 400

        if not all(
            np.isfinite(v) for v in valeurs_numeriques.values()
        ):
            return jsonify({
                "erreur": "Les valeurs numériques doivent être finies."
            }), 400

        if any(
            valeurs_numeriques[c] < 0
            for c in valeurs_numeriques
        ):
            return jsonify({
                "erreur": "Les valeurs numériques ne peuvent pas être négatives."
            }), 400

        # Préparer les données exactement comme à l'entraînement
        tache = pd.DataFrame([{
            **valeurs_numeriques,
            "rubrique": str(donnees["rubrique"]).strip(),
            "responsable": str(donnees["responsable"]).strip()
        }])

        X = preprocesseur.transform(tache)

        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=float)

        if X.shape[1] != len(w):
            return jsonify({
                "erreur": "Le nombre de variables préparées ne correspond pas au modèle.",
                "variables_preparees": int(X.shape[1]),
                "coefficients_modele": int(len(w))
            }), 500

        # Régression logistique
        z = float((X @ w + b)[0])

        # Sigmoïde stable numériquement
        if z >= 0:
            probabilite = 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            probabilite = exp_z / (1.0 + exp_z)

        risque = probabilite >= SEUIL

        return jsonify({
            "probabilite_retard": round(
                float(probabilite) * 100, 2
            ),
            "seuil": SEUIL,
            "classe_predite": int(risque),
            "niveau_risque": (
                "Risque de retard" if risque
                else "Risque de retard non détecté"
            ),
            "definition_cible": (
                "Retard d'au moins 2 jours"
            ),
            "avertissement": (
                "Modèle expérimental entraîné sur des données synthétiques. "
                "Résultat indicatif, à valider sur des données réelles."
            )
        })

    except Exception as e:
        app.logger.exception("Erreur pendant la prédiction")
        return jsonify({
            "erreur": "Erreur interne pendant la prédiction."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)