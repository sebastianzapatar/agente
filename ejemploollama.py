from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_URL="http://localhost:11434"

MODELO="llama3.1:8b"


llm=ChatOllama(
    model=MODELO,
    temperature=0.7,
    ollama_url=OLLAMA_URL
)

respuesta=llm.invoke("¿Cuál es la capital de Francia?")

print(respuesta.content)

#Con variables en el prompt
prompt=ChatPromptTemplate.from_messages([
    ("system","Eres un asistente que responde preguntas de {materia}. "
    "Responde en 2 oraciones"),
    ("human","{pregunta}")
])
respuesta=llm.invoke(prompt.format(materia="Programación",
                                    pregunta="¿Qué es una API REST?"))

print(respuesta.content)