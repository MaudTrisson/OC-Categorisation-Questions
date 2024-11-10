# OC-Categorisation-Questions

## Objectif du projet

Prédire une étiquette de catégorisation correspondante au titre d’un sujet proposé sur Stack Overflow

## Process

Récupération des données de Stackoverflow (notebook : Trisson_Maud_1_notebook_api_extract_112024)
Nettoyage et prétraitement des données (notebook : Trisson_Maud_2_notebook_nettoyage_exploratoire_112024)
Création, paramétrage et execution des modèles (notebooks : Trisson_Maud_3_notebook_approche_non_supervisée_112024, Trisson_Maud_4_notebook_approche_supervisée_classique_112024, Trisson_Maud_5_notebook_approche_supervisée_embedding_112024)
Enregistrement des données des modèles (Mlflow UI)
Déploiement d’une API (Application : dossier Trisson_Maud_6_FastAPI_app_112024 / Serveur AWS : http://15.237.192.126:8000/)
Automatisation du deploiement : Github actions (.github/workflows/deploy.yml)

