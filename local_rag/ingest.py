import os
import uuid
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from local_rag.config import config


def ingestar_documentos():

    if not os.path.exists(config.DOCS_DIR):
        os.makedirs(config.DOCS_DIR)
        print(f"El directorio {config.DOCS_DIR} ha sido creado. Por favor, coloca tus documentos allí y vuelve a ejecutar el script.")
        return 0
    loaders={
        ".txt": TextLoader,
        ".pdf": PyPDFLoader
    }
    documents=[]

    for filename in os.listdir(config.DOCS_DIR):
       file_path=os.path.join(config.DOCS_DIR, filename)
       ext=os.path.splitext(filename)[1]
       if ext in loaders:
           print("Cargando documento:", filename)
           loader_class=loaders[ext]
           loader=loader_class(file_path)
           documents.extend(loader.load())
       else:
           print(f"Archivo {filename} con extensión {ext} no soportado, se omitirá.")
    if not documents:
        print(f"No se encontraron documentos en {config.DOCS_DIR}. Por favor, agrega algunos documentos y vuelve a ejecutar el script.")
        return 0
    print(f"Total de documentos cargados: {len(documents)}")


    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, 
                                             chunk_overlap=200,
                                             add_start_index=True)

    chunks=text_splitter.split_documents(documents)

    print(f"Total de chunks generados: {len(chunks)}")

    embedding_model=OllamaEmbeddings(model=config.EMBEDDING_MODEL, 
                                     base_url=config.OLLAMA_URL)
    print("Generando embeddings")

    vector_store=PGVector(
        embeddings=embedding_model,
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_URI,
        use_jsonb=True
    )

    ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks,ids=ids)

    print("Proceso completado, los documentos han sido ingestado y almacenados en la base de datos.")
    return len(chunks)

if __name__=="__main__":
    ingestar_documentos()