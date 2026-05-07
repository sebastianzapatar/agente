FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar poetry
RUN pip install --no-cache-dir poetry

# Copiar archivos de dependencias
COPY pyproject.toml poetry.lock ./

# Configurar poetry para no usar virtualenvs (instalar globalmente en el contenedor)
RUN poetry config virtualenvs.create false

# Instalar dependencias
RUN poetry install --no-interaction --no-ansi --no-root

# Copiar el resto del código
COPY . .

# Instalar el paquete actual
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

# Iniciar la aplicación
CMD ["uvicorn", "local_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]