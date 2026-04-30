"""
Agente RAG con memoria conversacional.

Usa LangChain para:
  - Recuperar documentos relevantes de PGVector
  - Reformular preguntas considerando el historial
  - Responder SOLO con información del RAG
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

from local_rag.config import config


# ─────────────────────────────────────────────────────────────
# Almacén de sesiones de memoria (en memoria RAM)
# ─────────────────────────────────────────────────────────────
_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Obtiene o crea el historial de chat para una sesión."""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def get_all_sessions() -> list[str]:
    """Retorna todos los IDs de sesión activos."""
    return list(_session_store.keys())


def delete_session(session_id: str) -> bool:
    """Elimina el historial de una sesión."""
    if session_id in _session_store:
        del _session_store[session_id]
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Componentes del RAG
# ─────────────────────────────────────────────────────────────

def _build_retriever():
    """Construye el retriever conectado a PGVector."""
    embedding_model = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_URL
    )
    vector_store = PGVector(
        embeddings=embedding_model,
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_URI,
        use_jsonb=True
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 3 docs balancea calidad vs velocidad
    )


def _format_docs(docs) -> str:
    """Formatea los documentos recuperados en texto plano."""
    if not docs:
        return "No se encontró información relevante en los documentos."
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        # Extraer solo el nombre del archivo
        source = source.split("/")[-1] if "/" in source else source
        formatted.append(
            f"[Documento {i} — {source}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


# ─────────────────────────────────────────────────────────────
# Prompt del sistema — Solo responde con info del RAG
# ─────────────────────────────────────────────────────────────

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
    """
    Construye el agente RAG con memoria conversacional.
    
    Retorna un RunnableWithMessageHistory listo para invocar.
    """
    llm = ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0.3,   # Baja temperatura para respuestas más fieles al contexto
        num_predict=512,   # Limita tokens de respuesta (más rápido)
        num_ctx=2048       # Ventana de contexto reducida (más rápido)
    )

    retriever = _build_retriever()

    # ── Paso 1: Reformular pregunta con contexto del historial ──
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    # ── Paso 2: Prompt principal con contexto RAG ──
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def retrieve_and_respond(input_dict):
        """Pipeline completo: reformular → buscar → responder."""
        chat_history = input_dict.get("chat_history", [])
        user_input = input_dict["input"]

        # Si hay historial, reformular la pregunta
        if chat_history:
            standalone_question = contextualize_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
        else:
            standalone_question = user_input

        # Recuperar documentos relevantes
        docs = retriever.invoke(standalone_question)
        context = _format_docs(docs)

        # Generar respuesta
        response = (qa_prompt | llm | StrOutputParser()).invoke({
            "input": user_input,
            "context": context,
            "chat_history": chat_history
        })

        return response

    # ── Paso 3: Envolver con memoria ──
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(retrieve_and_respond)

    agent_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return agent_with_memory


# ─────────────────────────────────────────────────────────────
# Singleton del agente
# ─────────────────────────────────────────────────────────────
_agent = None


def get_agent():
    """Retorna la instancia singleton del agente."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


# ─────────────────────────────────────────────────────────────
# CLI interactivo para pruebas
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Agente RAG con Memoria")
    print("   Solo responde con información de tus documentos.")
    print("   Escribe 'salir' para terminar.")
    print("=" * 60)

    agent = get_agent()
    session_id = "cli-session"

    while True:
        try:
            pregunta = input("\n👤 Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 ¡Hasta luego!")
            break

        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("👋 ¡Hasta luego!")
            break

        print("🔍 Buscando en documentos...")
        respuesta = agent.invoke(
            {"input": pregunta},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\n🤖 Agente: {respuesta}")
