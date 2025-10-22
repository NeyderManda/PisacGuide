from llama_cpp import Llama
from langchain.vectorstores import Chroma
# IMPORTANTE: Importamos LlamaCppEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import sys

# Ruta a nuestro modelo GGUF (que se usará para TODO)
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DATA_PATH = "./datos_pisac.txt"

# --- 1. Cargar y Procesar Documentos ---
print("Cargando datos de Pisac...")
loader = TextLoader(DATA_PATH)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

print(f"Documentos cargados y divididos en {len(docs)} chunks.")

# --- 2. Crear Embeddings y VectorStore (¡EL CAMBIO!) ---
print("Cargando LlamaCppEmbeddings (usando Mistral GGUF)...")
# Usamos el mismo modelo GGUF para crear los embeddings.
# Esto elimina la dependencia de torch/sentence-transformers
embeddings = LlamaCppEmbeddings(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, # Usar la GPU para los embeddings
    n_batch=512,
    verbose=False
)

print("Creando base de datos vectorial (esto puede tardar un momento)...")
# Usamos Chroma como base de datos vectorial en memoria
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2}) # k=2 -> recuperar 2 chunks

print("Base de datos vectorial creada en memoria.")

# --- 3. Cargar el LLM (Mistral) ---
print("Cargando Mistral 7B (GGUF) para generación de chat...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, # Usar la GPU de 12GB
    n_ctx=2048,
    verbose=False
)
print("Modelo LLM cargado.")

# --- 4. Loop de Chat RAG ---

prompt_template = """
Contexto:
{context}

Pregunta:
{question}

Respuesta (basada únicamente en el contexto):
"""

while True:
    try:
        query = input("\n> (Escribe 'salir' para terminar)\n> Pregunta sobre Pisac: ")
        if query.lower() in ['salir', 'exit', 'quit']:
            break
        
        if not query:
            continue

        # 5. Recuperar (Retrieve)
        print("...Buscando en la base de datos...")
        context_docs = retriever.invoke(query)
        context_str = "\n\n".join([doc.page_content for doc in context_docs])

        # 6. Aumentar (Augment)
        prompt_lleno = prompt_template.format(context=context_str, question=query)

        # 7. Generar (Generate)
        print("...Generando respuesta con Mistral...")
        output = llm(
            prompt_lleno,
            max_tokens=150,
            stop=["Pregunta:", "\n"],
            echo=False
        )
        
        print("\nRespuesta de PisacGuide:")
        print(output['choices'][0]['text'].strip())

    except Exception as e:
        print(f"Error: {e}")
        break
    except KeyboardInterrupt:
        print("\nSaliendo...")
        break
        
print("Sprint 2 finalizado.")
