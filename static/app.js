// --- CONFIGURACIÓN DE LA PLANTILLA ---
// Esta es la única línea que podrías necesitar cambiar.
// '/chat' funciona cuando el frontend y el backend están en el mismo servidor (nuestro caso).
// Si despliegas el frontend en Netlify y el backend en Ngrok, pegarías la URL de Ngrok aquí.
const API_CHAT_URL = '/chat';
// ---------------------------------

const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');

// Frases en Quechua para identidad cultural
const quechuaPhrases = [
    "Tupananchiskama. (Hasta luego)",
    "Allinllachu. (¿Estás bien?)",
    "Sulpayki. (Gracias)",
    "Haku. (Vamos)",
    "Kusikumuni. (Me alegro)",
    "Rimaykullayki. (Te saludo)",
    "Munay. (Bonito/Lindo)",
    "Allin p'unchay. (Buenos días)"
];

function getRandomQuechuaPhrase() {
    const randomIndex = Math.floor(Math.random() * quechuaPhrases.length);
    return quechuaPhrases[randomIndex];
}

// Enviar mensaje al presionar Enter
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Enviar mensaje al hacer clic en el botón
sendButton.addEventListener('click', sendMessage);

function addMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll
}

function setLoading(isLoading) {
    if (isLoading) {
        // Añadir burbuja de "pensando..."
        const loadingBubble = document.createElement('div');
        loadingBubble.classList.add('message', 'bot', 'loading');
        loadingBubble.innerHTML = '<span></span><span></span><span></span>'; // Animación de puntos
        loadingBubble.id = 'loading-bubble';
        chatMessages.appendChild(loadingBubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } else {
        // Quitar burbuja de "pensando..."
        const loadingBubble = document.getElementById('loading-bubble');
        if (loadingBubble) {
            loadingBubble.remove();
        }
    }
}

async function sendMessage() {
    const question = chatInput.value.trim();
    if (question === '') return;

    // 1. Mostrar la pregunta del usuario
    addMessage(question, 'user');
    chatInput.value = '';
    setLoading(true);

    try {
        // 2. Enviar la pregunta al backend (FastAPI)
        const response = await fetch(API_CHAT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        });

        if (!response.ok) {
            throw new Error(`Error del servidor: ${response.statusText}`);
        }

        const data = await response.json();
        
        // 3. Quitar "pensando..." y mostrar la respuesta de la IA
        setLoading(false);
        
        // Añadir frase en Quechua al final
        const quechuaFooter = `\n\n✨ ${getRandomQuechuaPhrase()}`;
        addMessage(data.answer + quechuaFooter, 'bot');

    } catch (error) {
        console.error('Error al contactar la API:', error);
        setLoading(false);
        addMessage(`Error de conexión: No se pudo contactar al servidor. Revisa el log de Uvicorn y Ngrok. (${error.message})`, 'bot-error');
    }
}
