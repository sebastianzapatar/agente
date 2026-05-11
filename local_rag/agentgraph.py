from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from local_rag.config import Config

"""
todo
1. definir el estado
2. Componentes Compartidos del LLM, Retriever, Prompts
3.Definir los nodos del grafo, cada nodo es una función que recibe un estado y devuelve un nuevo estado
4. Construccion del grafo
5. gestion de sesiones y memoria
6. Para probar el main
"""
#1 Definicion del estado
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    standalone_question: str
    context: str
    response: str

#2 Componentes Compartidos
def _create_llm():
    """
    Crea una instancia de ChatOllama con la configuración especificada.
    Configuracion:
    - temperature : 0.3 respuesta mas deterministica
    -num_predict : 512 Limita la longitud de la respuesta
    -num_ctx : 2048 Limita el contexto total (entrada + respuesta) 
    """
    return ChatOllama(
        model=Config.OLLAMA_MODEL,
        temperature=0.3, #No es necesario en modelos nuevos
        num_predict=512,
        num_ctx=2048
    )
def _create_retriever():
    """
    Crea el retriever utilizando PGVector con la configuración especificada.
    Busca los 3 documentos más relevantes para la consulta dada.
    busqueda por similitud coseno sobre los embeddings 
    almacenados en la tabla 'documents' de la base de datos PostgreSQL.
    """
    embedding_model = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.OLLAMA_BASE_URL    
    )
    vector_store = PGVector(
        embeddings=embedding_model,
        collection_name=Config.COLLECTION_NAME,
        connection=Config.PG_URI,
        use_jsonb=True
    )
    return vector_store.as_retriever(
        search_kwargs={"k": 3},
        search_type="similarity"
    )
#Prompts del sistema
SYSTEM_PROMPT = """\
Eres un asistente de consulta documental. Tu ÚNICA función es responder \
preguntas usando EXCLUSIVAMENTE la información contenida en los documentos \
proporcionados como contexto.

REGLAS ESTRICTAS:
1. SOLO responde con información que esté en el contexto proporcionado.
2. Si la pregunta NO puede responderse con el contexto, di: \
"No tengo información sobre eso en los documentos disponibles."
3. NO inventes información ni uses conocimiento externo.
4. Cita la fuente del documento cuando sea posible.
5. Responde en español.
6. Si el usuario saluda, responde cordialmente pero recuérdale que \
solo puedes ayudar con información de los documentos cargados.

CONTEXTO DE LOS DOCUMENTOS:
{context}
"""

CONTEXTUALIZE_PROMPT = """\
Dado el historial de la conversación y la última pregunta del usuario, \
reformula la pregunta para que sea comprensible SIN el historial. \
NO respondas la pregunta, solo reformúlala si es necesario. \
Si ya es clara, devuélvela tal cual.
"""

#3 Definir los nodos del grafo
def reformular_pregunta(state:AgentState)->dict:
    #Nodo 1 reformular pregunta
    chat_history = state.get("chat_history", [])
    user_input=state["input"]

    #Si no hay historial, no es necesario reformular
    if not chat_history:
        return {"standalone_question": user_input}
    
    #Crear cadena de reformulación
    llm=_create_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        ("human", MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"))
    ])
    chain = prompt | llm | StrOutputParser()
    #Invocar la reformulación
    standalone=chain.invoke({
        "input":user_input,
        "chat_history":chat_history
    })
    return {"standalone_question": standalone}

def recuperar_documento(state:AgentState)->dict:
    #nodo2 recuperar documentos
    retriever=_create_retriever()
    question=state["standalone_question"]
    #Buscar documentos relevantes
    docs=retriever.invoke(question)
    #Formaterar documentos
    if not docs:
        context="No hay información relevante en los documentos."
    else:
        formatted=[]
        for i, doc in enumerate(docs,1):
            source=doc.metadata.get("source","desconocido")
            source=source.split("/")[-1] if "/" in source else source   #Extraer el nombre del archivo
            formatted.append(f"Documento {i} (Fuente: {source}):\n{doc.page_content}")
        context="\n\n".join(formatted)
    return {"context": context} #./nombre.pdf

def generar_respuesta(state:AgentState)->dict:
    #Nodo 3 generar respuesta
    llm=_create_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"))
    ])
    chain = prompt | llm | StrOutputParser()
    response=chain.invoke({
        "input": state["standalone_question"],
        "chat_history": state.get("chat_history", []),
        "context": state.get("context", "")
    })
    return {"response": response}