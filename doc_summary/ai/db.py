import os
import hashlib
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

load_dotenv()

# Define the persistent directory
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "assistant_db")
print(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.getenv("DB_DIR")
DATA_DIR = os.getenv("DATA_DIR")

def get_file_hash(filepath):
    """Generate a hash for the given file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_processed_files():
    """Load the hashes of files that have already been processed from a JSON file."""
    metadata_file = os.path.join(DATA_DIR, f"metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(processed_files):
    """Save the updated hashes of processed files to a JSON file."""
    metadata_file = os.path.join(DATA_DIR, f"metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(processed_files, f)

def generate_docs():
    already_processed_files = load_processed_files()
    loader = DirectoryLoader(DATA_DIR, glob=["**/*.xlsx", "**/*.md", "**/*.txt", "**/*.html", "**/*pdf"], recursive=True, show_progress=True)
    
    # Only load files that have not been processed yet
    all_docs = loader.load()
    new_docs = []
    new_processed_files = already_processed_files.copy()  # Create a copy to track new files

    for doc in all_docs:
        file_path = doc.metadata['source']
        file_hash = get_file_hash(file_path)
        
        if file_hash not in already_processed_files:
            new_docs.append(doc)
            new_processed_files[file_hash] = file_path  # Track this file as processed
        else:
            print(f"File {file_path} has already been processed. Skipping...")
    
    # Split the documents into chunks if there are new files
    if new_docs:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(new_docs), new_processed_files
    return [], new_processed_files


class DB():
    def __init__(self):
        if not os.path.exists(DB_DIR):
            raise ValueError(f"DB_DIR {DB_DIR} does not exist.")
        if not os.path.exists(DATA_DIR):
            raise ValueError(f"DATA_DIR {DATA_DIR} does not exist.")

        persist_directory = os.path.join(DB_DIR, COLLECTION_NAME)
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
        )
    def add_documents(self, docs):
        self.store.add_documents(docs)

    def add_document_from_file(self, filepath):
        loader = UnstructuredFileLoader(filepath)
        # Only load files that have not been processed yet
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.store.add_documents(text_splitter.split_documents(docs))
    
    def get_store(self):
        return self.store
    
db_obj = DB()

if __name__ == "__main__":
    print("--- Starting to process data folders ---")
    docs, new_processed_files = generate_docs()
    db_obj.get_store.add_documents(docs)
    save_processed_files(new_processed_files)
    print("--- Done ---")

