import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- Importaciones de LangChain (RAG) ---
from langchain.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from llama_cpp import Llama

# --- Variables Globales de Configuración ---
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DATA_PATH = "./data"
CHROMA_DB_PATH = "./chroma_db_persist"

# --- Variables Globales del Modelo (se cargan 1 vez) ---
llm_model: Optional[Llama] = None
vector_db: Optional[Chroma] = None

# --- Funciones de Inicialización (Sprint 3) ---

def initialize_llm():
    """Carga el modelo Mistral 7B en la VRAM."""
    print("Cargando modelo LLM en la GPU...")
    # n_gpu_layers=-1 significa "usar toda la GPU" (los 12GB)
    return Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=True
    )

def initialize_vector_db(llm_instance: Llama):
    """Carga o crea la base de datos vectorial."""
    print("Inicializando base de datos vectorial...")
    
    # 1. Crear el motor de embeddings (usando el LLM ya cargado)
    # Esta fue nuestra solución clave del Sprint 2
    embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH, n_gpu_layers=-1)
    
    # 2. Comprobar si la DB ya existe en disco
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Cargando base de datos existente desde: {CHROMA_DB_PATH}")
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        return db
    
    # 3. Si no existe, crearla desde cero
    print(f"Creando nueva base de datos desde: {DATA_PATH}")
    
    # Cargar todos los archivos .txt de la carpeta /data
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Dividir los documentos
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Crear y guardar la base de datos en disco
    print("Creando embeddings y guardando en disco...")
    db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=CHROMA_DB_PATH
    )
    print("¡Base de datos creada con éxito!")
    return db

# --- Inicialización de la API FastAPI ---

app = FastAPI(title="PisacGuide API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite CUALQUIER origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite TODOS los métodos (POST, GET)
    allow_headers=["*"],  # Permite CUALQUIER cabecera
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Servir el index.html en la raíz
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
# --- FIN DEL NUEVO CÓDIGO ---

@app.on_event("startup")
def on_startup():
    """
    Evento de inicio de FastAPI: Carga el modelo y la DB en memoria 1 SOLA VEZ.
    Esto cumple con el requisito de rendimiento (PGI-004).
    """
    global llm_model, vector_db
    
    # 1. Cargar el LLM (ocupa la VRAM)
    llm_model = initialize_llm()
    
    # 2. Cargar/Crear la Base Vectorial
    vector_db = initialize_vector_db(llm_model)
    
    print("--- ¡Servidor listo y modelo cargado! ---")

# --- Modelos de Datos (Pydantic) ---

class QueryRequest(BaseModel):
    """Define la estructura de la pregunta del usuario."""
    question: str

class QueryResponse(BaseModel):
    """Define la estructura de la respuesta del chatbot."""
    answer: str
    source_context: Optional[str] = None # Para depuración

# --- Endpoint de Chat (El corazón de la API) ---

@app.post("/chat", response_model=QueryResponse)
def handle_chat(query: QueryRequest):
    """
    Maneja una pregunta del usuario siguiendo el flujo RAG.
    """
    global llm_model, vector_db
    
    if llm_model is None or vector_db is None:
        return QueryResponse(answer="Error: El sistema se está iniciando, intenta en unos segundos.", source_context="Error")

    print(f"Recibida pregunta: {query.question}")
    
    # 1. Recuperar (Retrieve)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    context_docs = retriever.invoke(query.question)
    context_str = "\n\n".join([doc.page_content for doc in context_docs])
    
    # 2. Aumentar (Augment) - ¡AQUÍ ESTÁ LA NUEVA PERSONALIDAD!
    # Le damos un rol (Role Prompting) para que actúe como guía.
    prompt_template = f"""
    Instrucciones: Eres 'PisacGuide', un guía turístico oficial del Parque Arqueológico de Pisac.
    - Tu tono es amable, acogedor y profesional.
    - Responde SIEMPRE en español.
    - Si la pregunta es sobre saludos o cortesía, responde amablemente.
    - Basa tus respuestas factuales ÚNICAMENTE en el siguiente contexto proporcionado.
    - Si la respuesta no está en el contexto, di: "Lo siento, como guía virtual aún no tengo esa información específica, pero puedo ayudarte con horarios y rutas principales."
    - Usa frases cortas y claras, ideales para leer en un celular.

    Contexto del Parque:
    {context_str}
    
    Pregunta del Turista:
    {query.question}
    
    Respuesta de PisacGuide:
    """
    
    # 3. Generar (Generate)
    output = llm_model(
        prompt_template,
        max_tokens=300, # Aumentamos un poco para que pueda explayarse si es necesario
        stop=["Pregunta del Turista:", "Instrucciones:"],
        echo=False
    )
    
    answer = output['choices'][0]['text'].strip()
    print(f"Respuesta generada: {answer}")
    
    return QueryResponse(answer=answer, source_context=context_str)

@app.get("/")
async def read_index():
    """Sirve la página principal (el chat)."""
    return FileResponse('static/index.html')
