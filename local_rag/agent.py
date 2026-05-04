from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from local_rag.config import config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

#Se crea el almacen de las sesiones, en ram en este caso
_session_store:dict[str, ChatMessageHistory] = {}

def get_session_history(session_id:str)->ChatMessageHistory:
    #Vamos a obtener o crear el historial de mensajes para la sesión dada
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]

def get_all_sessions()->list[str]:
    #Obtenemos la lista de todas las sesiones activas
    return list(_session_store.keys())

def delete_session(session_id:str)->bool:
    #Eliminamos la sesión dada, si existe
    if session_id in _session_store:
        del _session_store[session_id]
        return True
    return False

#Acceder a los componente del RAG

def _build_retriever():
    """Devolvemos la información relevante de la base de datos para el mensaje dado"""
    embedding_model=OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, 
        base_url=config.OLLAMA_URL
    )
    vector_store=PGVector(
        embeddings=embedding_model,
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_URI,
        use_jsonb=True
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5}
    )

def _format_docs(docs)->str:
    """Formateamos los documentos recuperados para que sean legibles por el modelo de lenguaje"""
    if not docs:
        return "No se encontraron documentos relevantes."
    formatted = []
    for i, doc in enumerate(docs, 1):
        source=doc.metadata.get("source", "desconocida")
        source=source.split("/")[-1] if source else source
        formatted.append(f"Documento {i} (Fuente: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)

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

def build_agent():
    """Construimos el agente RAG con los componentes necesarios
    para responder preguntas usando los documentos almacenados."""
    llm=ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0.3,
        num_predict=512,
        num_ctx=2048
    )
    retriever=_build_retriever()
    contextualize_promt=ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    contextualize_chain=contextualize_promt | llm | StrOutputParser()

    qa_prompt=ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    def retrieve_and_response(input_dict):
        #Reformular la pregunta para que el agente pueda entenderla sin el historial
        #Buscar
        #Responder
        chat_history = input_dict.get("chat_history", [])
        user_input = input_dict["input"]
        #Si hay historial, reformulamos la pregunta
        if chat_history:
            standalone_question=contextualize_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
        else:
            standalone_question=user_input
        #Recuperar la informacion relevante usando el retriever
        docs=retriever.invoke(standalone_question)
        context=_format_docs(docs)
        #Generar la respuesta usando el prompt de QA
        response=(qa_prompt | llm | StrOutputParser()).invoke({
            "input": standalone_question,
            "chat_history": chat_history,
            "context": context
        })
        return response
    #Creamos la memoria, o envolvemos la memoria
    from langchain_core.runnables import RunnableLambda
    chain=RunnableLambda(retrieve_and_response)
    agent_with_memory=RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    return agent_with_memory
#Vamos crear una instancia global del agente para que pueda ser reutilizada en las consultas
#Usamos el patrón singleton para evitar crear múltiples instancias del agente y compartir la memoria de las sesiones
_agent  = None

def get_agent():
    #Retorna solo una instancia del agente, creando una nueva si no existe. Esto asegura que todas las consultas compartan la misma memoria de sesiones.
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


if __name__=="__main__":
    print("="*60)
    print("Agente RAG con memoria.")
    print("Solo responde usando la información de los documentos cargados.")
    print("Escribe 'salir', 'exit' o 'quit' para terminar la conversación.")
    print("="*60)
    agent = get_agent()
    session_id = "cli_session"  # Usamos una sesión fija para esta
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Terminando la conversación. ¡Hasta luego!")
            break
        response = agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print("Agente:", response)
