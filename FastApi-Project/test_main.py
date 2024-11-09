from fastapi.testclient import TestClient
from main import app  # Assurez-vous que le nom de l'application est bien 'app'

client = TestClient(app)

def test_predict_label():
    # Cas de test pour la prédiction d'étiquette avec un titre d'exemple
    response = client.post("/predict/", json={"title": "Exemple de sujet"})
    assert response.status_code == 200
    assert "label" in response.json()  # Vérifie si la réponse contient une clé 'label'