from dotenv import load_dotenv
import os
import sys
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore, auth
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

load_dotenv()

app = FastAPI(title="Backend IA Superviseur (Serverless Vercel)")

# Activer CORS pour permettre les appels depuis le Frontend Web et Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== INITIALISATION ROBUSTE GEMINI ====================
gemini_client = None
gemini_error = None
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if gemini_api_key:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=gemini_api_key)
        print("✅ Gemini AI initialisé")
    except Exception as e:
        gemini_error = str(e)
        print(f"⚠️ Erreur initialisation Gemini : {e}")
else:
    gemini_error = "GEMINI_API_KEY manquante dans les variables d'environnement Vercel"
    print("⚠️ GEMINI_API_KEY non configurée")

# ==================== INITIALISATION ROBUSTE FIREBASE ====================
db = None
firebase_error = None

def init_firebase():
    global db, firebase_error
    if db is not None:
        return db

    if firebase_admin._apps:
        db = firestore.client()
        return db

    firebase_credentials = None

    # 1. Depuis variable FIREBASE_SA_JSON (Vercel)
    sa_json = os.environ.get("FIREBASE_SA_JSON")
    if sa_json:
        try:
            # Nettoyage si encodé en base64 ou chaîne JSON
            if sa_json.startswith("ey"):  # Base64 pattern
                sa_json = base64.b64decode(sa_json).decode("utf-8")
            firebase_credentials = json.loads(sa_json)
        except Exception as e:
            print(f"⚠️ Erreur parsing FIREBASE_SA_JSON : {e}")

    # 2. Depuis fichier local (développement)
    if not firebase_credentials:
        key_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "./firebase-key.json")
        if os.path.exists(key_path):
            try:
                with open(key_path, "r", encoding="utf-8") as f:
                    firebase_credentials = json.load(f)
            except Exception as e:
                print(f"⚠️ Erreur lecture fichier clé : {e}")

    if firebase_credentials:
        try:
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase initialisé avec succès")
        except Exception as e:
            firebase_error = str(e)
            print(f"⚠️ Erreur init Firebase Admin : {e}")
    else:
        firebase_error = "Aucune clé Firebase trouvée (variable FIREBASE_SA_JSON ou fichier local)"
        print("⚠️ Firebase non initialisé (clé manquante)")

    return db

# Initialisation au démarrage
try:
    init_firebase()
except Exception as e:
    firebase_error = str(e)

# ==================== UTILITAIRES ====================
def extraire_notes(reponses: dict) -> list:
    return [
        v for v in (reponses or {}).values()
        if isinstance(v, int) and not isinstance(v, bool)
    ]

async def obtenir_conseil_gemini(moyenne: float) -> str:
    if not gemini_client:
        return generer_conseil_fallback(moyenne)

    model_names = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for model_name in model_names:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=f"Donne un conseil d'optimisation en 20 mots maximum pour un magasin avec une note moyenne de {moyenne:.1f}/5. Sois précis et actionnable."
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"⚠️ Erreur avec modèle {model_name} : {e}")
            continue

    return generer_conseil_fallback(moyenne)

def generer_conseil_fallback(moyenne: float) -> str:
    if moyenne >= 4.5:
        return "Excellent facing et disponibilité optimale. Maintenez le niveau de réassort actuel."
    elif moyenne >= 3.5:
        return "Disponibilité correcte. Priorisez le réassort des linéaires frais avant l'affluence de 10h."
    elif moyenne >= 2.5:
        return "Ruptures constatées en rayon. Vérifiez d'urgence les stocks en réserve et le balisage prix."
    else:
        return "Niveau critique de ruptures. Intervention commerciale urgente requise auprès du point de vente."

# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Microservice Python IA Superviseur (Serverless Vercel)",
        "diagnostics": {
            "gemini_ready": gemini_client is not None,
            "gemini_error": gemini_error,
            "firebase_ready": db is not None,
            "firebase_error": firebase_error,
        }
    }

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/reponses")
async def recevoir_reponse(payload: dict, authorization: str = Header(None)):
    firestore_db = init_firebase()
    if not firestore_db:
        raise HTTPException(
            status_code=503,
            detail=f"Service Firebase non disponible. Erreur: {firebase_error}"
        )

    try:
        reponse_id = payload.get("reponse_id")
        magasin_id = payload.get("magasin_id")

        if not reponse_id or not magasin_id:
            raise HTTPException(status_code=400, detail="reponse_id et magasin_id sont obligatoires")

        # Idempotence : vérifier si déjà traitée
        existing = firestore_db.collection("reponses").document(reponse_id).get()
        if existing.exists:
            stats = firestore_db.collection("statistiques").document(magasin_id).get()
            stats_data = stats.to_dict() if stats.exists else {}
            return JSONResponse(status_code=409, content={
                "status": "already_processed",
                "moyenne": stats_data.get("moyenne", 0),
                "conseil": stats_data.get("conseil_ia", ""),
            })

        # Sauvegarder la réponse
        firestore_db.collection("reponses").document(reponse_id).set(payload)

        # Calcul de la moyenne avec pandas
        docs = list(firestore_db.collection("reponses").where("magasin_id", "==", magasin_id).stream())
        total_reponses = len(docs)
        toutes_les_notes = []
        for d in docs:
            toutes_les_notes.extend(extraire_notes(d.to_dict().get("reponses")))

        df = pd.DataFrame({"note": toutes_les_notes})
        moyenne = float(df["note"].mean()) if not df.empty else 0.0

        # Obtenir le conseil IA
        conseil = await obtenir_conseil_gemini(moyenne)

        # Mettre à jour les statistiques
        firestore_db.collection("statistiques").document(magasin_id).set({
            "moyenne": moyenne,
            "conseil_ia": conseil,
            "total_reponses": total_reponses
        }, merge=True)

        return {
            "status": "ok",
            "moyenne": moyenne,
            "conseil": conseil,
            "total_reponses": total_reponses
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur lors du traitement de la réponse : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)