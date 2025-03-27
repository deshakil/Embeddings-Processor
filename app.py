from flask import Flask, request, jsonify
import os
import json
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

app = Flask(__name__)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

METADATA_CONTAINER = "weez-files-metadata"
EMBEDDINGS_CONTAINER = "weez-files-embeddings"

AZURE_OPENAI_ENDPOINT = "https://weez-openai-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
EMBEDDING_MODEL = "text-embedding-3-large"

def fetch_user_blobs(container_client, user_id):
    user_prefix = f"{user_id}/"
    return container_client.list_blobs(name_starts_with=user_prefix)

def read_blob_content(container_client, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall().decode('utf-8')

def compute_embeddings_with_azure_openai(metadata):
    text = " ".join(str(value) for value in metadata.values())
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    embeddings = response.data[0].embedding
    return embeddings

def upload_embeddings(blob_client, file_name, embeddings, file_path):
    embeddings_data = {
        "file_name": file_name,
        "embeddings": embeddings,
        "file_path": file_path
    }
    embeddings_json = json.dumps(embeddings_data)
    blob_client.upload_blob(embeddings_json, overwrite=True)

def process_user_metadata_to_embeddings(user_id):
    metadata_container_client = blob_service_client.get_container_client(METADATA_CONTAINER)
    embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
    if not embeddings_container_client.exists():
        blob_service_client.create_container(EMBEDDINGS_CONTAINER)
    blobs = fetch_user_blobs(metadata_container_client, user_id)
    processed_files = []
    failed_files = []
    for blob in blobs:
        try:
            metadata_content = read_blob_content(metadata_container_client, blob.name)
            metadata = json.loads(metadata_content)
            file_path = metadata.get("file_path", "unknown_path")
            embeddings = compute_embeddings_with_azure_openai(metadata)
            embeddings_blob_client = embeddings_container_client.get_blob_client(blob.name)
            upload_embeddings(embeddings_blob_client, blob.name, embeddings, file_path)
            processed_files.append(blob.name)
        except Exception as e:
            failed_files.append({"blobremotename": blob.name, "error": str(e)})
    return {
        "processed_files": processed_files,
        "failed_files": failed_files
    }

@app.route('/process_embeddings', methods=['POST'])
def process_embeddings():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id in request"}), 400
    try:
        result = process_user_metadata_to_embeddings(user_id)
        return jsonify({"message": "Processing completed", "result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_single_embedding', methods=['POST'])
def process_single_embedding():
    data = request.get_json()
    user_id = data.get('user_id')
    blob_name = data.get('blob_name')
    if not user_id or not blob_name:
        return jsonify({"error": "Missing user_id or blob_name in request"}), 400
    
    metadata_container_client = blob_service_client.get_container_client(METADATA_CONTAINER)
    embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
    
    if not embeddings_container_client.exists():
        blob_service_client.create_container(EMBEDDINGS_CONTAINER)
    
    try:
        # Ensure blob_name has the proper user_id prefix if not provided
        if not blob_name.startswith(f"{user_id}/"):
            blob_name = f"{user_id}/{blob_name}"
        
        # Read metadata
        metadata_content = read_blob_content(metadata_container_client, blob_name)
        metadata = json.loads(metadata_content)
        
        # Extract file path
        file_path = metadata.get("file_path", "unknown_path")
        
        # Compute embeddings
        embeddings = compute_embeddings_with_azure_openai(metadata)
        
        # Upload embeddings
        embeddings_blob_client = embeddings_container_client.get_blob_client(blob_name)
        upload_embeddings(embeddings_blob_client, blob_name, embeddings, file_path)
        
        return jsonify({
            "status": "success",
            "blob_name": blob_name,
            "message": "Embeddings generated and stored successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "blob_name": blob_name,
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Changed port to avoid conflict with generateMetadata.py
