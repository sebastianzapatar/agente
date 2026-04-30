"""
Agente RAG con LangGraph — Implementación con grafo de estados.

Este módulo implementa un agente conversacional usando LangGraph,
que permite definir el flujo como un grafo dirigido con nodos y aristas.

Arquitectura del Grafo:
  ┌─────────────┐     ┌───────────────┐     ┌──────────────┐
  │ reformular  │────▶│   recuperar   │────▶│  responder   │
  └─────────────┘     └───────────────┘     └──────────────┘
        ▲                                          │
        │              Estado Global               │
        └──────────────────────────────────────────┘

Ventajas sobre el agente con LangChain puro:
  - Flujo visual y explícito como grafo
  - Estado tipado y predecible con TypedDict
  - Fácil de extender con nuevos nodos (validación, herramientas, etc.)
  - Mejor depuración: se puede inspeccionar el estado en cada nodo
"""

from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from local_rag.config import config


# ═══════════════════════════════════════════════════════════════
# 1. DEFINICIÓN DEL ESTADO
# ═══════════════════════════════════════════════════════════════
# El estado es el "corazón" de LangGraph. Es un diccionario tipado
# que fluye entre todos los nodos del grafo. Cada nodo puede leer
# y modificar el estado.

class AgentState(TypedDict):
    """
    Estado global del agente que se comparte entre todos los nodos.
    
    Atributos:
        input: La pregunta original del usuario.
        chat_history: Lista de mensajes previos (memoria).
        standalone_question: Pregunta reformulada sin depender del historial.
        context: Texto de los documentos recuperados del vector store.
        response: La respuesta final generada por el LLM.
    """
    input: str                              # Pregunta del usuario
    chat_history: list[BaseMessage]         # Historial de la conversación
    standalone_question: str                # Pregunta reformulada
    context: str                            # Documentos recuperados
    response: str                           # Respuesta final


# ═══════════════════════════════════════════════════════════════
# 2. COMPONENTES COMPARTIDOS (LLM, Retriever, Prompts)
# ═══════════════════════════════════════════════════════════════

def _create_llm():
    """
    Crea la instancia del modelo de lenguaje (Ollama local).
    
    Configuración:
      - temperature=0.3: Respuestas más deterministas y fieles al contexto.
      - num_predict=512: Limita la longitud de respuesta para mayor velocidad.
      - num_ctx=2048: Ventana de contexto reducida para procesamiento más rápido.
    """
    return ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0.3,
        num_predict=512,
        num_ctx=2048
    )


def _create_retriever():
    """
    Crea el retriever conectado a PostgreSQL con pgvector.
    
    Busca los 3 documentos más similares a la consulta usando
    búsqueda por similitud coseno sobre los embeddings almacenados.
    """
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
        search_kwargs={"k": 3}
    )


# Prompts del sistema
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


# ═══════════════════════════════════════════════════════════════
# 3. NODOS DEL GRAFO
# ═══════════════════════════════════════════════════════════════
# Cada nodo es una función que recibe el estado, lo procesa,
# y retorna las actualizaciones al estado.

def reformular_pregunta(state: AgentState) -> dict:
    """
    Nodo 1: Reformular la pregunta.
    
    Si hay historial de conversación, usa el LLM para reformular
    la pregunta del usuario de manera que sea autocontenida
    (comprensible sin necesidad del historial).
    
    Si NO hay historial, usa la pregunta tal cual (ahorra una
    llamada al LLM → más rápido).
    
    Entrada del estado: input, chat_history
    Salida al estado: standalone_question
    """
    chat_history = state.get("chat_history", [])
    user_input = state["input"]

    # Optimización: si no hay historial, no reformular
    if not chat_history:
        return {"standalone_question": user_input}

    # Crear cadena de reformulación
    llm = _create_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    chain = prompt | llm | StrOutputParser()

    # Invocar la reformulación
    standalone = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    return {"standalone_question": standalone}


def recuperar_documentos(state: AgentState) -> dict:
    """
    Nodo 2: Recuperar documentos relevantes.
    
    Usa la pregunta reformulada para buscar los documentos
    más similares en la base de datos vectorial (pgvector).
    Los documentos se formatean en texto plano con su fuente.
    
    Entrada del estado: standalone_question
    Salida al estado: context
    """
    retriever = _create_retriever()
    question = state["standalone_question"]

    # Buscar documentos similares
    docs = retriever.invoke(question)

    # Formatear documentos recuperados
    if not docs:
        context = "No se encontró información relevante en los documentos."
    else:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconocido")
            source = source.split("/")[-1] if "/" in source else source
            formatted.append(f"[Documento {i} — {source}]\n{doc.page_content}")
        context = "\n\n".join(formatted)

    return {"context": context}


def generar_respuesta(state: AgentState) -> dict:
    """
    Nodo 3: Generar la respuesta final.
    
    Usa el contexto de los documentos recuperados y el historial
    de la conversación para generar una respuesta que se limite
    estrictamente a la información disponible.
    
    Entrada del estado: input, context, chat_history
    Salida al estado: response
    """
    llm = _create_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "input": state["input"],
        "context": state["context"],
        "chat_history": state.get("chat_history", [])
    })

    return {"response": response}


# ═══════════════════════════════════════════════════════════════
# 4. CONSTRUCCIÓN DEL GRAFO
# ═══════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Construye el grafo del agente RAG con LangGraph.
    
    Flujo del grafo:
      START → reformular → recuperar → responder → END
    
    Cada nodo procesa una parte del pipeline:
      1. reformular: Hace la pregunta autocontenida
      2. recuperar: Busca documentos relevantes
      3. responder: Genera la respuesta final
    
    Retorna:
        El grafo compilado listo para ejecutar con .invoke()
    """
    # Crear el grafo con el tipo de estado definido
    graph = StateGraph(AgentState)

    # ── Agregar nodos al grafo ──
    graph.add_node("reformular", reformular_pregunta)
    graph.add_node("recuperar", recuperar_documentos)
    graph.add_node("responder", generar_respuesta)

    # ── Definir las aristas (conexiones entre nodos) ──
    graph.add_edge(START, "reformular")           # Inicio → Reformular
    graph.add_edge("reformular", "recuperar")     # Reformular → Recuperar
    graph.add_edge("recuperar", "responder")      # Recuperar → Responder
    graph.add_edge("responder", END)              # Responder → Fin

    # ── Compilar el grafo ──
    return graph.compile()


# ═══════════════════════════════════════════════════════════════
# 5. GESTIÓN DE SESIONES Y MEMORIA
# ═══════════════════════════════════════════════════════════════

# Almacén de historiales de chat por sesión (en memoria RAM)
_session_store: dict[str, list[BaseMessage]] = {}

# Singleton del grafo compilado
_compiled_graph = None


def get_graph():
    """Retorna la instancia singleton del grafo compilado."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def get_session_history(session_id: str) -> list[BaseMessage]:
    """Obtiene o crea el historial de chat para una sesión."""
    if session_id not in _session_store:
        _session_store[session_id] = []
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


def chat(question: str, session_id: str = "default") -> str:
    """
    Función principal para interactuar con el agente.
    
    Ejecuta el grafo completo y gestiona la memoria de la sesión.
    
    Args:
        question: La pregunta del usuario.
        session_id: Identificador de la sesión para mantener memoria.
    
    Returns:
        La respuesta del agente como string.
    """
    graph = get_graph()
    history = get_session_history(session_id)

    # Ejecutar el grafo con el estado inicial
    result = graph.invoke({
        "input": question,
        "chat_history": history,
        "standalone_question": "",
        "context": "",
        "response": ""
    })

    # Guardar en el historial de la sesión
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=result["response"]))

    return result["response"]


# ═══════════════════════════════════════════════════════════════
# 6. CLI INTERACTIVO PARA PRUEBAS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Agente RAG con LangGraph")
    print("   Grafo: reformular → recuperar → responder")
    print("   Solo responde con información de tus documentos.")
    print("   Escribe 'salir' para terminar.")
    print("=" * 60)

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

        print("🔍 Ejecutando grafo: reformular → recuperar → responder...")
        respuesta = chat(pregunta, session_id)
        print(f"\n🤖 Agente: {respuesta}")
