import os
import hashlib
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
data_dir = os.path.join(current_dir, "data")
metadata_dir = os.path.join(current_dir, "metadata")  # Directory to store processed file metadata

os.makedirs(metadata_dir, exist_ok=True)


def get_file_hash(filepath):
    """Generate a hash for the given file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_processed_files(data_name):
    """Load the hashes of files that have already been processed from a JSON file."""
    metadata_file = os.path.join(metadata_dir, f"{data_name}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}


def save_processed_files(data_name, processed_files):
    """Save the updated hashes of processed files to a JSON file."""
    metadata_file = os.path.join(metadata_dir, f"{data_name}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(processed_files, f)


def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        store = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. Checking for updates...")
        store = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
        store.add_documents(docs)
        print(f"--- Updated vector store {store_name} with new documents ---")


def create_docs(notes_dir, processed_files):
    loader = DirectoryLoader(notes_dir, glob=["**/*.xlsx", "**/*.md", "**/*.txt", "**/*.html", "**/*pdf"], recursive=True, show_progress=True)
    
    # Only load files that have not been processed yet
    all_docs = loader.load()
    new_docs = []
    new_processed_files = processed_files.copy()  # Create a copy to track new files

    for doc in all_docs:
        file_path = doc.metadata['source']
        file_hash = get_file_hash(file_path)
        
        if file_hash not in processed_files:
            new_docs.append(doc)
            new_processed_files[file_hash] = file_path  # Track this file as processed
        else:
            print(f"File {file_path} has already been processed. Skipping...")
    
    # Split the documents into chunks if there are new files
    if new_docs:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(new_docs), new_processed_files
    else:
        return [], new_processed_files


def create_db(data_name):
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    notes_dir = os.path.join(data_dir, data_name)
    if not os.path.exists(notes_dir):
        raise FileNotFoundError(f"The directory {notes_dir} does not exist. Please check the path.")
    
    # Load already processed files
    processed_files = load_processed_files(data_name)
    
    docs, new_processed_files = create_docs(notes_dir, processed_files)
    
    if docs:
        create_vector_store(docs, embedding_function, data_name)
        save_processed_files(data_name, new_processed_files)
    else:
        print(f"No new documents to add for {data_name}.")


def process_data_folders():
    # Automatically detect all subdirectories in the data folder
    data_folders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    
    # create_db("chirag")
    for data_name in data_folders:
        try:
            print(f"\n--- Processing folder: {data_name} ---")
            create_db(data_name)
        except Exception as e:
            print(f"Error processing {data_name}: {e}")


def get_db(data_name):
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persistent_directory = os.path.join(db_dir, data_name)
    return Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )

if __name__ == "__main__":
    print("--- Starting to process data folders ---")
    process_data_folders()
    print("--- Done ---")
