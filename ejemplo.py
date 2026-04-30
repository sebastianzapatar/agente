import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada.")

from langchain_openai import ChatOpenAI #Escogemos el modelo de lenguaje que queremos usar, en este caso gpt-3.5-turbo
from langchain_core.prompts import ChatPromptTemplate #Plantilla de prompts
from langchain_core.output_parsers import StrOutputParser #Parser para salida de texto  

llm=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

respuesta=llm.invoke("¿Cuál es la capital de Francia?")


print(respuesta.content)


"""
Se pueden colocar variables dentro del prompt, 
para eso se usa el ChatPromptTemplate, por ejemplo:
"""
prompt=ChatPromptTemplate.from_messages([
    ("system","Eres un asistente que responde preguntas de {materia}."),
    ("human","{pregunta}")
])

mensaje_formateado=prompt.format(materia="Inteligencia Artificial",
                                  pregunta="Que es un transformer?")

respuesta=llm.invoke(mensaje_formateado)
print(respuesta.content)