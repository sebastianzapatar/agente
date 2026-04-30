import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    PG_URI=os.getenv("DATABASE_URL")
    print("PG_URI:", PG_URI)
    COLLECTION_NAME="documentos_local"
    CHAT_MODEL="llama3.1:8b"
    EMBEDDING_MODEL="nomic-embed-text"
    OLLAMA_URL=os.getenv("OLLAMA_URL")
    print("OLLAMA_URL:", OLLAMA_URL)
    DOCS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)), "documentos")

config=Config()



