from azure.storage.blob import BlobServiceClient
import pickle
import pandas as pd
import io
import os
from dotenv import load_dotenv

load_dotenv() 

BLOB_ACCOUNT_NAME = os.getenv("BLOB_ACCOUNT_NAME")
BLOB_ACCOUNT_KEY = os.getenv("BLOB_ACCOUNT_KEY")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "mycontentdata")

blob_service_client = BlobServiceClient(account_url=f"https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net", credential=BLOB_ACCOUNT_KEY)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def download_blob(file_name):
    blob_client = container_client.get_blob_client(file_name)
    download_stream = blob_client.download_blob()
    return download_stream.readall()

def upload_blob(file_name, data):
    blob_client = container_client.get_blob_client(file_name)
    blob_client.upload_blob(data, overwrite=True)

def load_model():
    model_data = download_blob("svdpp_model.pkl")
    return pickle.loads(model_data)

def load_articles_df():
    csv_data = download_blob("articles_metadata.csv")
    return pd.read_csv(io.BytesIO(csv_data))

def load_embeddings():
    embeddings_data = download_blob("articles_embeddings.pickle")
    return pickle.loads(embeddings_data)

def save_model(model):
    model_data = pickle.dumps(model)
    upload_blob("svdpp_model.pkl", model_data)

def save_articles_df(df):
    csv_data = df.to_csv(index=False).encode()
    upload_blob("articles_metadata.csv", csv_data)

def save_embeddings(embeddings):
    embeddings_data = pickle.dumps(embeddings)
    upload_blob("articles_embeddings.pickle", embeddings_data)

def load_clicks_df():
    # Télécharger le fichier des clics depuis Azure Blob Storage
    csv_data = download_blob("clicks_sample.csv")
    clicks_df = pd.read_csv(io.BytesIO(csv_data))
    return clicks_df


