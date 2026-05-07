"""
API REST con FastAPI para el agente RAG.

Endpoints:
  POST /chat              — Enviar pregunta al agente
  POST /ingest            — Ingestar documentos al vector store
  GET  /sessions          — Listar sesiones activas
  GET  /sessions/{id}     — Ver historial de una sesión
  DELETE /sessions/{id}   — Eliminar historial de una sesión
  GET  /health            — Estado del servicio
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from local_rag.agent import get_agent, get_session_history, get_all_sessions, delete_session
from local_rag.ingest import ingestar_documentos


# ─────────────────────────────────────────────────────────────
# Lifespan — inicializa el agente al arrancar
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga el agente al iniciar el servidor."""
    print("🚀 Inicializando agente RAG...")
    get_agent()
    print("✅ Agente listo.")
    yield
    print("👋 Servidor apagado.")


# ─────────────────────────────────────────────────────────────
# App FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="🧠 SinRodilla RAG API",
    description="API para consultar documentos usando RAG con memoria conversacional.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Modelos Pydantic
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Modelo de entrada para el chat."""
    pregunta: str = Field(..., min_length=1, description="La pregunta a realizar")
    session_id: str = Field(default="default", description="ID de la sesión de conversación")

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
    """Modelo de respuesta del chat."""
    respuesta: str
    session_id: str


class MessageOut(BaseModel):
    """Modelo para un mensaje del historial."""
    rol: str
    contenido: str


class SessionHistoryResponse(BaseModel):
    """Modelo de respuesta del historial de sesión."""
    session_id: str
    mensajes: list[MessageOut]
    total_mensajes: int


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Envía una pregunta al agente RAG.
    
    El agente buscará en los documentos cargados y responderá
    SOLO con información encontrada en ellos. La conversación
    se mantiene en memoria por `session_id`.
    """
    agent = get_agent()

    try:
        respuesta = agent.invoke(
            {"input": request.pregunta},
            config={"configurable": {"session_id": request.session_id}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del agente: {str(e)}")

    return ChatResponse(
        respuesta=respuesta,
        session_id=request.session_id
    )


@app.post("/ingest", tags=["Documentos"])
async def ingest():
    """
    Ingesta de documentos.

    Lee todos los archivos PDF y TXT de la carpeta `documentos/`,
    los divide en chunks, genera embeddings con Ollama
    y los almacena en la base de datos PostgreSQL con pgvector.
    """
    try:
        total_chunks = ingestar_documentos()
        return {
            "mensaje": "Ingesta completada exitosamente.",
            "chunks_generados": total_chunks,
            "ok": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la ingesta: {str(e)}")


@app.get("/sessions", tags=["Sesiones"])
async def list_sessions():
    """Lista todas las sesiones de conversación activas."""
    sessions = get_all_sessions()
    return {
        "sesiones": sessions,
        "total": len(sessions)
    }


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse, tags=["Sesiones"])
async def get_history(session_id: str):
    """Obtiene el historial de mensajes de una sesión."""
    if session_id not in get_all_sessions():
        raise HTTPException(status_code=404, detail=f"Sesión '{session_id}' no encontrada.")

    history = get_session_history(session_id)
    mensajes = []
    for msg in history.messages:
        rol = "usuario" if msg.type == "human" else "agente"
        mensajes.append(MessageOut(rol=rol, contenido=msg.content))

    return SessionHistoryResponse(
        session_id=session_id,
        mensajes=mensajes,
        total_mensajes=len(mensajes)
    )


@app.delete("/sessions/{session_id}", tags=["Sesiones"])
async def clear_session(session_id: str):
    """Elimina el historial de una sesión de conversación."""
    if delete_session(session_id):
        return {"mensaje": f"Sesión '{session_id}' eliminada.", "ok": True}
    raise HTTPException(status_code=404, detail=f"Sesión '{session_id}' no encontrada.")


@app.get("/health", tags=["Sistema"])
async def health_check():
    """Verifica que el servicio esté funcionando."""
    return {
        "status": "ok",
        "servicio": "SinRodilla RAG API",
        "modelo_chat": "llama3.1:8b",
        "modelo_embeddings": "nomic-embed-text"
    }