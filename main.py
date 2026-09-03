from dotenv import load_dotenv
import os
import sys

load_dotenv()

# ==================== DIAGNOSTIC (pour Vercel) ====================
print("=== DIAGNOSTIC START ===")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("GEMINI_API_KEY present:", bool(os.environ.get("GEMINI_API_KEY")))
print("FIREBASE_SA_JSON present:", bool(os.environ.get("FIREBASE_SA_JSON")))
if os.environ.get("FIREBASE_SA_JSON"):
    print("FIREBASE_SA_JSON (first 200 chars):", os.environ["FIREBASE_SA_JSON"][:200])
print("=== DIAGNOSTIC END ===")

# Vérification des clés obligatoires
if not os.environ.get("GEMINI_API_KEY"):
    raise Exception("❌ GEMINI_API_KEY non défini dans l'environnement")

import json
import firebase_admin
import pandas as pd
from firebase_admin import credentials, firestore, auth
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from google import genai

# ==================== INITIALISATION GEMINI ====================
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    print("✅ Client Gemini initialisé")
except Exception as e:
    print(f"❌ Erreur initialisation Gemini : {e}")
    raise

# ==================== INITIALISATION FIREBASE ====================
firebase_credentials = None

# 1. Essayer la variable d'environnement (pour Vercel)
firebase_json_str = os.environ.get("FIREBASE_SA_JSON")
if firebase_json_str:
    print("🔑 Tentative chargement depuis FIREBASE_SA_JSON")
    try:
        firebase_credentials = json.loads(firebase_json_str)
        print("✅ Firebase chargé depuis variable d'environnement")
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON dans FIREBASE_SA_JSON : {e}")
        firebase_credentials = None

# 2. Fallback : fichier local (pour développement)
if not firebase_credentials:
    firebase_cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "./firebase-key.json")
    if os.path.exists(firebase_cred_path):
        print(f"📄 Firebase : chargement depuis {firebase_cred_path}")
        with open(firebase_cred_path, 'r') as f:
            firebase_credentials = json.load(f)
        print("✅ Firebase chargé depuis fichier")
    else:
        raise Exception("❌ Aucune clé Firebase trouvée (ni FIREBASE_SA_JSON, ni fichier local)")

# Initialisation de l'application Firebase
try:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialisé avec succès")
except Exception as e:
    print(f"❌ Erreur initialisation Firebase : {e}")
    raise

# ==================== APPLICATION FASTAPI ====================
app = FastAPI()

# ==================== FONCTION DE VÉRIFICATION TOKEN ====================
def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Token manquant")
    token = authorization.replace("Bearer ", "")
    try:
        return auth.verify_id_token(token)
    except Exception:
        raise HTTPException(401, "Token invalide")

def extraire_notes(reponses: dict) -> list:
    """Le mobile envoie reponses = {question_id: valeur}. Seules les questions
    de type 'note' produisent un entier 1-5 (les 'oui_non' sont des bool,
    exclus ici car bool est une sous-classe de int en Python)."""
    return [
        v for v in (reponses or {}).values()
        if isinstance(v, int) and not isinstance(v, bool)
    ]

# ==================== ENDPOINT PRINCIPAL ====================
@app.post("/reponses")
async def recevoir_reponse(payload: dict, authorization: str = Header(None)):
    try:
        # (en développement : désactiver la vérification)
        print("🔓 Vérification token désactivée (test local)")
        decoded = {"uid": "test-user"}  # valeur factice

        reponse_id = payload["reponse_id"]
        magasin_id = payload["magasin_id"]

        # Idempotence : si cette réponse a déjà été traitée (retry réseau côté
        # mobile), on ne recalcule rien et on ne rappelle pas Gemini.
        existing = db.collection("reponses").document(reponse_id).get()
        if existing.exists:
            print(f"↩️ Réponse {reponse_id} déjà traitée, idempotent")
            stats = db.collection("statistiques").document(magasin_id).get()
            stats_data = stats.to_dict() if stats.exists else {}
            return JSONResponse(status_code=409, content={
                "status": "already_processed",
                "moyenne": stats_data.get("moyenne", 0),
                "conseil": stats_data.get("conseil_ia", ""),
            })

        # Sauvegarder la réponse brute
        db.collection("reponses").document(reponse_id).set(payload)
        print(f"✅ Réponse {reponse_id} sauvegardée")

        # Récupérer toutes les réponses du magasin (une seule lecture) et en extraire les notes
        docs = list(db.collection("reponses").where("magasin_id", "==", magasin_id).stream())
        total_reponses = len(docs)
        toutes_les_notes = []
        for d in docs:
            toutes_les_notes.extend(extraire_notes(d.to_dict().get("reponses")))

        df = pd.DataFrame({"note": toutes_les_notes})
        moyenne = float(df["note"].mean()) if not df.empty else 0.0
        print(f"📊 {len(toutes_les_notes)} notes trouvées sur {total_reponses} réponses, moyenne={moyenne:.2f}")

        # Appel à Gemini
        conseil = await obtenir_conseil_gemini(moyenne)

        # Sauvegarder les statistiques
        db.collection("statistiques").document(magasin_id).set({
            "moyenne": moyenne,
            "conseil_ia": conseil,
            "total_reponses": total_reponses
        }, merge=True)

        return {"status": "ok", "moyenne": moyenne, "conseil": conseil}

    except Exception as e:
        print(f"❌ Erreur : {e}")
        raise HTTPException(500, f"Erreur interne: {str(e)}")

# ==================== FONCTION D'APPEL À GEMINI ====================
async def obtenir_conseil_gemini(moyenne: float) -> str:
    """Interroge Gemini pour obtenir un conseil, avec fallback."""
    model_names = [
        "gemini-3.6-flash",   # Dernier modèle (juillet 2026)
        "gemini-2.0-flash",   # Stable
        "gemini-3.0-flash",   # Alternative
    ]
    for model_name in model_names:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=f"Donne un conseil d'optimisation en 20 mots maximum pour un magasin avec une note moyenne de {moyenne:.1f}/5. Sois précis et actionnable."
            )
            conseil = response.text
            print(f"🤖 Conseil Gemini (via {model_name}) : {conseil}")
            return conseil
        except Exception as e:
            print(f"⚠️ Erreur avec {model_name} : {e}")
            continue
    # Fallback si tous les modèles échouent
    return generer_conseil_fallback(moyenne)

# ==================== FONCTION DE FALLBACK ====================
def generer_conseil_fallback(moyenne: float) -> str:
    if moyenne >= 4.5:
        return "Service premium maintenu !"
    elif moyenne >= 3.5:
        return "Améliorez l'accueil client."
    elif moyenne >= 2.5:
        return "Revoyez la qualité du service."
    else:
        return "Plan d'action urgent requis."

# ==================== ENDPOINTS DE TEST ====================
@app.get("/")
async def root():
    return {"message": "API Backend Enquête - OK"}

@app.get("/test-firebase")
async def test_firebase():
    try:
        db.collection("test").document("test").set({"test": "ok"})
        return {"status": "Firebase OK"}
    except Exception as e:
        return {"status": "Firebase ERREUR", "error": str(e)}