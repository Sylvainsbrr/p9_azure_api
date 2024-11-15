import azure.functions as func
import datetime
import json
import logging
import json
import pickle
import pandas as pd
import numpy as np
import logging
import random
import time
from utils import load_model, load_articles_df, load_embeddings, save_articles_df, save_embeddings, load_clicks_df


app = func.FunctionApp()

# Charger les articles et les embeddings existants
model = load_model()
articles_df = load_articles_df()
embeddings = load_embeddings()
clicks_df = load_clicks_df()


# Liste des utilisateurs
users = set(clicks_df['user_id'].unique())

# Fonction pour obtenir le titre d'un article par ID
def get_article_title(article_id):
    return f"Article {article_id}"  # Utilise l'ID comme titre fictif


# Route pour ajouter un nouvel utilisateur
@app.function_name(name="AddUserFunction")
@app.route(route="add_user", methods=["POST"])
def add_user_function(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        user_id = req_body.get('user_id')

        if user_id is None:
            return func.HttpResponse("Invalid request: 'user_id' is required.", status_code=400)

        if user_id in users:
            return func.HttpResponse("User already exists.", status_code=400)

        # Ajouter le nouvel utilisateur
        users.add(user_id)
        return func.HttpResponse(f"User {user_id} added successfully.", status_code=201)

    except Exception as e:
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)


# Fonction pour obtenir les articles populaires par catégorie
def get_popular_articles():
    # Compter les clics pour chaque article
    click_counts = clicks_df.groupby('click_article_id').size().reset_index(name='click_count')
    click_counts.sort_values(by='click_count', ascending=False, inplace=True)

    # Sélectionner les 5 articles les plus cliqués
    top_articles = click_counts.head(5)['click_article_id'].tolist()

    return top_articles


# Route pour obtenir des recommandations basées sur le filtrage collaboratif ou des articles populaires
@app.function_name(name="RecommendFunction")
@app.route(route="recommend", methods=["GET"])
def recommend_function(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.params.get('user_id')
    
    if not user_id:
        return func.HttpResponse("Please provide a user_id in the query string", status_code=400)

    try:
        user_id = int(user_id)
    except ValueError:
        return func.HttpResponse("Invalid user_id. Must be an integer.", status_code=400)

    if user_id not in users:
        return func.HttpResponse("User not found. Please add the user first.", status_code=404)

    # Si l'utilisateur est nouveau (aucune interaction), recommander les articles populaires
    if user_id not in clicks_df['user_id'].values:
        top_articles = get_popular_articles()
        response = {
            "user_id": user_id,
            "recommendations": [
                {
                    "article_id": article_id,
                    "title": get_article_title(article_id),
                    "score": "N/A (Popular Article)"
                }
                for article_id in top_articles
            ]
        }
        return func.HttpResponse(json.dumps(response), mimetype="application/json", status_code=200)

    # Recommandations basées sur le filtrage collaboratif
    cf_recommendations = [
        (article_id, model.predict(user_id, article_id).est)
        for article_id in articles_df['article_id']
    ]
    cf_recommendations.sort(key=lambda x: x[1], reverse=True)
    top_cf_recommendations = cf_recommendations[:5]

    response = {
        "user_id": user_id,
        "recommendations": [
            {
                "article_id": article_id,
                "title": get_article_title(article_id),
                "score": score
            }
            for article_id, score in top_cf_recommendations
        ]
    }

    return func.HttpResponse(json.dumps(response), mimetype="application/json", status_code=200)

# Route pour ajouter un nouvel article avec un embedding aléatoire
@app.function_name(name="AddArticleFunction")
@app.route(route="add_article", methods=["POST"])
def add_article_function(req: func.HttpRequest) -> func.HttpResponse:
    try:
        global articles_df
        global embeddings

        logging.info("Début de l'ajout de l'article.")

        # Extraire les données de l'article depuis la requête JSON
        req_body = req.get_json()
        new_article_id = req_body.get('article_id')
        category_id = req_body.get('category_id', 0)
        publisher_id = req_body.get('publisher_id', 0)
        words_count = req_body.get('words_count', 0)
        created_at_ts = int(time.time() * 1000)

        if new_article_id is None:
            logging.error("ID de l'article manquant.")
            return func.HttpResponse("Invalid request: 'article_id' is required.", status_code=400)

        # Vérifier si l'article existe déjà
        if new_article_id in articles_df['article_id'].values:
            logging.warning("L'article existe déjà.")
            return func.HttpResponse("Article already exists.", status_code=400)

        logging.info("Sélection d'un embedding aléatoire.")
        random_embedding = embeddings[random.randint(0, len(embeddings) - 1)]
        
        # Créer un nouvel article
        new_article = pd.DataFrame({
            'article_id': [new_article_id],
            'category_id': [category_id],
            'created_at_ts': [created_at_ts],
            'publisher_id': [publisher_id],
            'words_count': [words_count]
        })

        logging.info("Ajout de l'article au DataFrame.")
        articles_df = pd.concat([articles_df, new_article], ignore_index=True)

        # Ajouter l'embedding à la liste des embeddings
        embeddings = np.vstack([embeddings, random_embedding])

        logging.info("Sauvegarde des modifications.")
        articles_df.to_csv("articles_metadata.csv", index=False)
        with open("articles_embeddings.pickle", "wb") as f:
            pickle.dump(embeddings, f)

        response = {
            "message": "Article added successfully.",
            "article_id": new_article_id,
            "category_id": category_id,
            "created_at_ts": created_at_ts,
            "publisher_id": publisher_id,
            "words_count": words_count
        }

        logging.info("Article ajouté avec succès.")
        return func.HttpResponse(json.dumps(response), mimetype="application/json", status_code=201)

    except Exception as e:
        logging.error(f"Une erreur est survenue : {str(e)}")
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)
    
    