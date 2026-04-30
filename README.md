<div align="center">

# 🧠 SinRodilla — RAG Local con Ollama

<img src="docs/mascot.png" alt="SinRodilla Mascot" width="250"/>

*Tu asistente de IA local que responde preguntas basándose en tus propios documentos.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama)](https://ollama.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

---

</div>

## 📋 Descripción

**SinRodilla** es un pipeline de **Retrieval-Augmented Generation (RAG)** que funciona **100% local**. Permite cargar documentos (PDF y TXT), generar embeddings vectoriales y realizar consultas inteligentes usando modelos de lenguaje ejecutados localmente con Ollama.

> 🔒 **Sin dependencias en la nube** — Tus datos nunca salen de tu máquina.

---

## 🏗️ Arquitectura

```
📁 sinrodilla/
├── 📁 documentos/            # 📄 Coloca aquí tus PDFs y TXTs
├── 📁 local_rag/             # 🧠 Módulo principal del RAG
│   ├── config.py             #    ⚙️  Configuración centralizada
│   ├── ingest.py             #    📥 Ingesta y vectorización de documentos
│   ├── agent.py              #    🤖 Agente conversacional (WIP)
│   └── api.py                #    🌐 API REST con FastAPI (WIP)
├── 📁 src/sinrodilla/        # 📦 Paquete principal
├── 📁 tests/                 # 🧪 Tests
├── ejemplo.py                # 💡 Ejemplo con OpenAI
├── ejemploollama.py          # 💡 Ejemplo con Ollama local
├── compose.yml               # 🐳 Docker Compose (PostgreSQL + Ollama)
├── pyproject.toml            # 📦 Dependencias (Poetry)
└── .env                      # 🔐 Variables de entorno
```

---

## 🔄 Flujo del Pipeline

```mermaid
graph LR
    A[📄 Documentos<br/>PDF / TXT] -->|Carga| B[📥 Loader<br/>LangChain]
    B -->|Split| C[✂️ Chunks<br/>1000 chars]
    C -->|Embeddings| D[🧮 Ollama<br/>nomic-embed-text]
    D -->|Almacena| E[(🐘 PostgreSQL<br/>pgvector)]
    E -->|Búsqueda<br/>Semántica| F[🤖 LLM<br/>llama3.1:8b]
    F --> G[💬 Respuesta]

    style A fill:#f9f,stroke:#333,color:#000
    style E fill:#4169E1,stroke:#333,color:#fff
    style F fill:#10b981,stroke:#333,color:#fff
    style G fill:#f59e0b,stroke:#333,color:#000
```

---

## 🚀 Inicio Rápido

### 1. Prerrequisitos

| Herramienta | Versión | Enlace |
|---|---|---|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| Docker & Docker Compose | Última | [docker.com](https://docker.com) |
| Poetry | ≥ 2.0 | [python-poetry.org](https://python-poetry.org) |

### 2. Clonar e instalar dependencias

```bash
git clone <tu-repo-url>
cd sinrodilla
poetry install
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/vectordb
OLLAMA_URL=http://localhost:11434
OPENAI_API_KEY=sk-...  # Solo si usas el ejemplo con OpenAI
```

### 4. Levantar servicios con Docker

```bash
docker compose up -d
```

Esto levanta automáticamente:

| Servicio | Puerto | Descripción |
|---|---|---|
| **PostgreSQL + pgvector** | `5432` | Base de datos vectorial |
| **Ollama** | `11434` | Servidor de modelos locales |
| **Ollama Init** | — | Descarga `llama3.1:8b` y `nomic-embed-text` |

> ⏳ La primera vez tarda unos minutos mientras descarga los modelos (~5 GB total).

### 5. Agregar documentos

Coloca tus archivos `.pdf` y `.txt` en la carpeta `documentos/`:

```bash
cp mi_documento.pdf documentos/
cp mis_notas.txt documentos/
```

### 6. Ejecutar la ingesta

```bash
poetry run python -m local_rag.ingest
```

Salida esperada:

```
Cargando documento: mi_documento.pdf
Cargando documento: mis_notas.txt
Total de documentos cargados: 5
Total de chunks generados: 29
Generando embeddings
Proceso completado, los documentos han sido ingestado y almacenados en la base de datos.
```

---

## 🛠️ Tecnologías

| Tecnología | Rol |
|---|---|
| **LangChain** | Orquestación del pipeline RAG |
| **Ollama** | Ejecución local de LLMs y embeddings |
| **llama3.1:8b** | Modelo de chat (generación de respuestas) |
| **nomic-embed-text** | Modelo de embeddings (vectorización) |
| **PostgreSQL + pgvector** | Almacenamiento y búsqueda vectorial |
| **Poetry** | Gestión de dependencias |
| **Docker Compose** | Orquestación de servicios |
| **FastAPI** | API REST (en desarrollo) |
| **python-dotenv** | Carga de variables desde `.env` |

---

## 📦 Dependencias Principales

```toml
[project.dependencies]
langchain = ">=1.2.15"
langchain-ollama = ">=1.1.0"
langchain-postgres = ">=0.0.17"
langchain-community = ">=0.4.1"
langchain-text-splitters = ">=1.1.2"
psycopg = ">=3.3.3"
fastapi = ">=0.136.1"
uvicorn = ">=0.46.0"
pypdf = ">=6.10.2"
python-dotenv = ">=1.2.2"
```

---

## 🗄️ Comandos Útiles

### Base de datos

```bash
# Ver colecciones almacenadas
docker exec rag-postgres psql -U postgres -d vectordb \
  -c "SELECT name FROM langchain_pg_collection;"

# Contar embeddings por colección
docker exec rag-postgres psql -U postgres -d vectordb \
  -c "SELECT c.name, COUNT(e.id) as total
      FROM langchain_pg_collection c
      LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
      GROUP BY c.name;"

# Limpiar todos los datos
docker exec rag-postgres psql -U postgres -d vectordb \
  -c "TRUNCATE langchain_pg_embedding, langchain_pg_collection CASCADE;"
```

### Ollama

```bash
# Ver modelos disponibles
docker exec rag-ollama ollama list

# Descargar un modelo adicional
docker exec rag-ollama ollama pull mistral:7b

# Probar un modelo
docker exec rag-ollama ollama run llama3.1:8b "Hola, ¿cómo estás?"
```

### Docker

```bash
# Levantar servicios
docker compose up -d

# Ver logs
docker compose logs -f

# Detener servicios
docker compose down

# Detener y borrar volúmenes (⚠️ borra datos)
docker compose down -v
```

---

## 💡 Ejemplos

### Ejemplo con Ollama (local)

```bash
poetry run python ejemploollama.py
```

### Ejemplo con OpenAI (requiere API key)

```bash
poetry run python ejemplo.py
```

---

## 🗺️ Roadmap

- [x] Ingesta de documentos PDF y TXT
- [x] Generación de embeddings con Ollama
- [x] Almacenamiento vectorial en PostgreSQL
- [x] Búsqueda por similitud semántica
- [ ] Agente conversacional con memoria
- [ ] API REST con FastAPI
- [ ] Frontend web para consultas
- [ ] Soporte para más formatos (DOCX, Markdown)

---

## 👤 Autor

**Sebastián Zapata**
📧 pansezapata@gmail.com

---

<div align="center">

*Hecho con 💜 y mucho ☕ — Sin rodilla, sin límites.*

</div>
