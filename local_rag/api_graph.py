"""
API REST con FastAPI para el agente RAG basado en LangGraph.

Este módulo expone el agente LangGraph como una API REST,
permitiendo interactuar con el RAG a través de endpoints HTTP.

Endpoints:
  POST   /chat              — Enviar pregunta al agente LangGraph
  POST   /ingest            — Ingestar documentos al vector store
  GET    /sessions          — Listar sesiones activas
  GET    /sessions/{id}     — Ver historial de una sesión
  DELETE /sessions/{id}     — Eliminar historial de una sesión
  GET    /health            — Estado del servicio

Diferencia con api.py:
  - api.py usa el agente basado en LangChain (RunnableWithMessageHistory)
  - api_graph.py usa el agente basado en LangGraph (StateGraph)
  
  LangGraph ofrece:
    - Flujo explícito como grafo de estados
    - Mayor control sobre cada paso del pipeline
    - Más fácil de extender con nodos adicionales
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage

# Importar el agente LangGraph y la función de ingesta
from local_rag.agent_graph import (
    get_graph,
    chat as agent_chat,
    get_session_history,
    get_all_sessions,
    delete_session
)
from local_rag.ingest import ingestar_documentos


# ─────────────────────────────────────────────────────────────
# Lifespan — Inicializa el grafo al arrancar el servidor
# ─────────────────────────────────────────────────────────────
# El lifespan es un context manager que se ejecuta al iniciar
# y al apagar el servidor. Lo usamos para precargar el grafo
# y evitar que la primera petición sea lenta.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga el grafo LangGraph al iniciar el servidor."""
    print("🚀 Inicializando agente RAG con LangGraph...")
    get_graph()  # Compila el grafo la primera vez
    print("✅ Grafo compilado y listo.")
    yield
    print("👋 Servidor apagado.")


# ─────────────────────────────────────────────────────────────
# Configuración de la aplicación FastAPI
# ─────────────────────────────────────────────────────────────
# Se configura el título, descripción y versión que aparecen
# en la documentación automática de Swagger UI (/docs).
app = FastAPI(
    title="🧠 SinRodilla RAG API — LangGraph",
    description=(
        "API para consultar documentos usando RAG con LangGraph.\n\n"
        "**Flujo del grafo:** reformular → recuperar → responder"
    ),
    version="2.0.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────
# Middleware CORS
# ─────────────────────────────────────────────────────────────
# Permite que cualquier frontend (React, Vue, etc.) se conecte
# a esta API sin restricciones de origen cruzado.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En producción, limitar a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Modelos Pydantic — Validación de datos de entrada/salida
# ─────────────────────────────────────────────────────────────
# Pydantic valida automáticamente los datos de las peticiones
# y genera la documentación del esquema en Swagger.

class ChatRequest(BaseModel):
    """Modelo de entrada para el endpoint de chat."""
    pregunta: str = Field(
        ...,
        min_length=1,
        description="La pregunta a realizar al agente"
    )
    session_id: str = Field(
        default="default",
        description="ID de la sesión. Misma sesión = misma memoria."
    )

    # Ejemplo para la documentación Swagger
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pregunta": "¿De qué tratan los documentos?",
                    "session_id": "usuario-1"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Modelo de respuesta del endpoint de chat."""
    respuesta: str
    session_id: str


class MessageOut(BaseModel):
    """Modelo para representar un mensaje del historial."""
    rol: str            # "usuario" o "agente"
    contenido: str      # El texto del mensaje


class SessionHistoryResponse(BaseModel):
    """Modelo de respuesta para el historial de una sesión."""
    session_id: str
    mensajes: list[MessageOut]
    total_mensajes: int


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Envía una pregunta al agente RAG con LangGraph.
    
    El flujo interno del grafo es:
      1. **Reformular**: Si hay historial, reformula la pregunta
      2. **Recuperar**: Busca documentos similares en pgvector
      3. **Responder**: Genera respuesta SOLO con el contexto encontrado
    
    La memoria se mantiene por `session_id`: misma sesión = misma conversación.
    """
    try:
        # agent_chat() ejecuta el grafo completo y gestiona la memoria
        respuesta = agent_chat(request.pregunta, request.session_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error del agente LangGraph: {str(e)}"
        )

    return ChatResponse(
        respuesta=respuesta,
        session_id=request.session_id
    )


@app.post("/ingest", tags=["Documentos"])
async def ingest():
    """
    Ingesta de documentos al vector store.
    
    Lee todos los archivos PDF y TXT de la carpeta `documentos/`,
    los divide en chunks de 1000 caracteres, genera embeddings
    con nomic-embed-text (Ollama) y los almacena en PostgreSQL.
    """
    try:
        total_chunks = ingestar_documentos()
        return {
            "mensaje": "Ingesta completada exitosamente.",
            "chunks_generados": total_chunks,
            "ok": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la ingesta: {str(e)}"
        )


@app.get("/sessions", tags=["Sesiones"])
async def list_sessions():
    """Lista todas las sesiones de conversación activas en memoria."""
    sessions = get_all_sessions()
    return {
        "sesiones": sessions,
        "total": len(sessions)
    }


@app.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    tags=["Sesiones"]
)
async def get_history(session_id: str):
    """
    Obtiene el historial completo de mensajes de una sesión.
    
    Cada mensaje tiene un rol ('usuario' o 'agente') y su contenido.
    """
    if session_id not in get_all_sessions():
        raise HTTPException(
            status_code=404,
            detail=f"Sesión '{session_id}' no encontrada."
        )

    history = get_session_history(session_id)
    mensajes = []
    for msg in history:
        # Determinar si el mensaje es del usuario o del agente
        rol = "usuario" if isinstance(msg, HumanMessage) else "agente"
        mensajes.append(MessageOut(rol=rol, contenido=msg.content))

    return SessionHistoryResponse(
        session_id=session_id,
        mensajes=mensajes,
        total_mensajes=len(mensajes)
    )


@app.delete("/sessions/{session_id}", tags=["Sesiones"])
async def clear_session(session_id: str):
    """
    Elimina el historial de una sesión de conversación.
    
    Esto "reinicia" la memoria del agente para esa sesión.
    """
    if delete_session(session_id):
        return {"mensaje": f"Sesión '{session_id}' eliminada.", "ok": True}
    raise HTTPException(
        status_code=404,
        detail=f"Sesión '{session_id}' no encontrada."
    )


@app.get("/health", tags=["Sistema"])
async def health_check():
    """
    Verifica que el servicio esté funcionando correctamente.
    
    Retorna información sobre el servicio, los modelos en uso
    y la versión del motor (LangGraph).
    """
    return {
        "status": "ok",
        "servicio": "SinRodilla RAG API — LangGraph",
        "motor": "LangGraph (StateGraph)",
        "modelo_chat": config.CHAT_MODEL,
        "modelo_embeddings": config.EMBEDDING_MODEL,
        "flujo": "reformular → recuperar → responder"
    }
