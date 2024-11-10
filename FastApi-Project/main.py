from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib

# Charger le modèle LDA et le CountVectorizer
model = joblib.load('countvectorizer-lda-fit-on-title-and-transform-on-title-body.pkl')
vectorizer = joblib.load('countvectorizer-lda-fit-on-title-and-transform-on-title-body_vectorizer.pkl')  # Charger le CountVectorizer utilisé lors de l'entraînement

topic_labels = {
    0: "'value', 'int', 'std', 'node', 'const', 'return', 'array', 'expo', 'list', 'char'",
    1: "'div', 'class', 'button', 'id', 'page', 'html', 'text', 'style', 'script', 'li'",
    2: "'app', 'error', 'server', 'http', 'using', 'client', 'service', 'api', 'use', 'run'",
    3: "'user', 'model', 'name', 'form', 'image', 'class', 'import', 'request', 'data', 'post'",
    4: "'file', 'lib', 'package', 'python', 'line', 'error', 'py', 'module', 'project', 'build'",
    5: "'file', 'write', 'function', 'code', 'wait', 'operation', 'ctrl', 'stat', 'name', 'document'",
    6: "'string', 'id', 'data', 'public', 'return', 'new', 'get', 'type', 'value', 'null'",
    7: "'android', 'const', 'import', 'color', 'react', 'text', 'component', 'app', 'self', 'layout'",
    8: "'data', 'df', 'column', 'value', 'output', 'self', 'row', 'model', 'like', 'input'",
    9: "'java', 'org', 'com', 'springframework', 'version', 'spring', 'dependency', 'jar', 'import', 'core'",
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