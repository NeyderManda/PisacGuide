from llama_cpp import Llama

# Ruta al modelo que descargaremos
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

print("Cargando el modelo... (Esto puede tardar varios minutos)")

# Configuración de la GPU: n_gpu_layers=-1 significa "usar toda la GPU"
# Esto es lo que probará la VRAM de 12GB.
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, 
    n_ctx=2048,       # Contexto de 2048 tokens
    verbose=True
)

print("Modelo cargado. Generando respuesta...")

# Prueba de inferencia
prompt = "Pregunta: ¿Qué es el Inti Raymi en Cusco?"

output = llm(
    prompt, 
    max_tokens=100, 
    stop=["Pregunta:", "\n"], 
    echo=False
)

print("\nRespuesta del modelo:")
print(output['choices'][0]['text'].strip())
print("\n¡Prueba del Sprint 1 completada con éxito!")
