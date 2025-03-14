from flask import Flask, request, jsonify
import os
import json
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

# Initialize Flask app
app = Flask(__name__)

# Initialize Azure Blob Service Client
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# Containers
METADATA_CONTAINER = "weez-files-metadata"
EMBEDDINGS_CONTAINER = "weez-files-embeddings"

# Azure OpenAI settings for embeddings
AZURE_OPENAI_ENDPOINT = "https://weez-openai-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
EMBEDDING_MODEL = "text-embedding-3-large"

def fetch_user_blobs(container_client, user_id):
    """
    Fetch all blobs from the metadata container that belong to the specified user_id.
    """
    user_prefix = f"{user_id}/"  # Use user_id as a prefix for user-specific files
    return container_client.list_blobs(name_starts_with=user_prefix)

def read_blob_content(container_client, blob_name):
    """
    Stream content of a blob directly from Azure.
    """
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall().decode('utf-8')

def compute_embeddings_with_azure_openai(metadata):
    """
    Compute embeddings for the metadata values using Azure OpenAI's embeddings model.
    """
    # Flatten metadata values into a single string
    text = " ".join(str(value) for value in metadata.values())

    # Create Azure OpenAI client
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    # Generate embeddings using Azure OpenAI
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    # Extract the embedding from the response
    embeddings = response.data[0].embedding

    # Close the client
    client.close()

    return embeddings

def upload_embeddings(blob_client, file_name, embeddings, file_path):
    """
    Upload embeddings as a JSON file to the target container.
    Store the embeddings inside a folder named after the user.
    """
    # Prepare the embeddings data, including the file path
    embeddings_data = {
        "file_name": file_name,
        "embeddings": embeddings,
        "file_path": file_path  # Add the file path here
    }

    embeddings_json = json.dumps(embeddings_data)
    blob_client.upload_blob(embeddings_json, overwrite=True)

def process_user_metadata_to_embeddings(user_id):
    """
    Main function to process metadata and store embeddings for a specific user.
    """
    # Initialize container clients
    metadata_container_client = blob_service_client.get_container_client(METADATA_CONTAINER)
    embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)

    # Ensure the target container exists
    if not embeddings_container_client.exists():
        blob_service_client.create_container(EMBEDDINGS_CONTAINER)

    # Fetch all user-specific metadata blobs
    blobs = fetch_user_blobs(metadata_container_client, user_id)

    processed_files = []
    failed_files = []

    for blob in blobs:
        try:
            # Step 1: Read metadata directly from blob
            metadata_content = read_blob_content(metadata_container_client, blob.name)
            metadata = json.loads(metadata_content)

            # Extract the file path from the metadata (assuming it exists in the metadata)
            file_path = metadata.get("file_path", "unknown_path")  # Default to "unknown_path" if not present

            # Step 2: Compute embeddings using Azure OpenAI
            embeddings = compute_embeddings_with_azure_openai(metadata)

            # Step 3: Upload embeddings to the target container with user_id in the blob path
            embeddings_blob_client = embeddings_container_client.get_blob_client(blob.name)
            upload_embeddings(embeddings_blob_client, blob.name, embeddings, file_path)

            processed_files.append(blob.name)
        except Exception as e:
            failed_files.append({"blob_name": blob.name, "error": str(e)})

    return {
        "processed_files": processed_files,
        "failed_files": failed_files
    }

@app.route('/process_embeddings', methods=['POST'])
def process_embeddings():
    """
    API endpoint to process embeddings for a specific user.
    """
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
def process_single_metadata_to_embedding(user_id, blob_name):
    """
    Process a single metadata blob to embedding for a specific user.
    """
    # Initialize container clients
    metadata_container_client = blob_service_client.get_container_client(METADATA_CONTAINER)
    embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)

    # Ensure the target container exists
    if not embeddings_container_client.exists():
        blob_service_client.create_container(EMBEDDINGS_CONTAINER)

    try:
        # Ensure blob_name has the proper user_id prefix
        if not blob_name.startswith(f"{user_id}/"):
            blob_name = f"{user_id}/{blob_name}"
            
        # Step 1: Read metadata directly from blob
        metadata_content = read_blob_content(metadata_container_client, blob_name)
        metadata = json.loads(metadata_content)
        
        # Extract the file path from the metadata
        file_path = metadata.get("file_path", "unknown_path")
        
        # Step 2: Compute embeddings using Azure OpenAI
        embeddings = compute_embeddings_with_azure_openai(metadata)
        
        # Step 3: Upload embeddings to the target container
        embeddings_blob_client = embeddings_container_client.get_blob_client(blob_name)
        upload_embeddings(embeddings_blob_client, blob_name, embeddings, file_path)
        
        return {
            "status": "success",
            "blob_name": blob_name
        }
    except Exception as e:
        return {
            "status": "error",
            "blob_name": blob_name,
            "message": str(e)
        }

if __name__ == '__main__':
    app.run(debug=True)
