import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

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
# Montar la carpeta 'static' para servir HTML/CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    Maneja una pregunta del usuario.
    1. Busca contexto en ChromaDB.
    2. Construye un prompt.
    3. Genera una respuesta con el LLM.
    """
    global llm_model, vector_db
    
    print(f"Recibida pregunta: {query.question}")
    
    # 1. Recuperar (Retrieve) - Usamos .invoke() (lo que aprendimos ayer)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # k=2 chunks
    context_docs = retriever.invoke(query.question)
    context_str = "\n\n".join([doc.page_content for doc in context_docs])
    
    # 2. Aumentar (Augment) - Creamos el prompt
    prompt_template = f"""
    Contexto:
    {context_str}
    
    Pregunta:
    {query.question}
    
    Respuesta (basada únicamente en el contexto):
    """
    
    # 3. Generar (Generate) - Usamos el modelo cargado en memoria
    output = llm_model(
        prompt_template,
        max_tokens=250,
        stop=["Pregunta:"],
        echo=False
    )
    
    answer = output['choices'][0]['text'].strip()
    print(f"Respuesta generada: {answer}")
    
    return QueryResponse(answer=answer, source_context=context_str)

@app.get("/")
async def read_index():
    """Sirve la página principal (el chat)."""
    return FileResponse('static/index.html')
