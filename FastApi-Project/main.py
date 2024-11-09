from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib

# Charger le modèle LDA et le CountVectorizer
model = joblib.load('countvectorizer-lda-fit-on-title-and-transform-on-title-body.pkl')
vectorizer = joblib.load('countvectorizer-lda-fit-on-title-and-transform-on-title-body_vectorizer.pkl')  # Charger le CountVectorizer utilisé lors de l'entraînement

topic_labels = {
    0: "Sport",
    1: "Politique",
    2: "Technologie",
    3: "Santé",
    4: "Sport",
    5: "Politique",
    6: "Technologie",
    7: "Santé",
    8: "Santé",
    9: "Santé",
    10: "Santé",
}

# Création de l'application FastAPI
app = FastAPI()

# Monter le répertoire 'static' pour servir les fichiers CSS, JS, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Créer un modèle Pydantic pour recevoir les données
class TitleInput(BaseModel):
    title: str

# Créer une route POST pour la prédiction
@app.post("/predict/")
async def predict_label(input: TitleInput):
    title = input.title  # Récupérer le titre de la requête
    
    # Transformer le titre pour qu'il soit compatible avec votre modèle
    vectorized_title = vectorizer.transform([title])  # Utiliser CountVectorizer pour transformer le titre
    
    # Effectuer la prédiction avec le modèle
    prediction = model.transform(vectorized_title)
    topic = prediction.argmax()  # Trouver le thème dominant (selon la méthode utilisée)

    # Récupérer le label correspondant à l'étiquette prédite
    label = topic_labels.get(topic, "Inconnu")  # Retourne "Inconnu" si l'étiquette n'existe pas
    
    # Retourner l'étiquette prédite et son numéro
    return {"topic_number": int(topic), "label": label}

# Route GET pour servir la page HTML
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)