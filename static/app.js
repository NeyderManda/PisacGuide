
// Espera a que todo el HTML esté cargado
document.addEventListener("DOMContentLoaded", () => {
    
    // Obtener los elementos del DOM
    const chatWindow = document.getElementById("chat-window");
    const chatInput = document.getElementById("chat-input");
    const sendButton = document.getElementById("send-button");
    const statusLight = document.getElementById("status-light");

    // --- Función para enviar el mensaje ---
    const sendMessage = async () => {
        const messageText = chatInput.value.trim();
        
        if (messageText === "") return; // No enviar mensajes vacíos

        // 1. Mostrar el mensaje del usuario en el chat
        addMessage(messageText, "user");
        chatInput.value = ""; // Limpiar el input

        // 2. Comprobar Modo Offline (Requisito PGI-007)
        if (!navigator.onLine) {
            addMessage("Modo Offline: No puedo conectarme al servidor de IA.", "bot");
            return;
        }

        // 3. Mostrar un indicador de "escribiendo..."
        addMessage("...", "bot-typing");

        try {
            // 4. Llamar a nuestra API (Sprint 3)
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ question: messageText }),
            });

            // 5. Quitar el indicador "escribiendo..."
            removeTypingIndicator();

            if (!response.ok) {
                throw new Error("Error en la respuesta de la API");
            }

            const data = await response.json();
            
            // 6. Mostrar la respuesta del Bot
            addMessage(data.answer, "bot");

        } catch (error) {
            console.error("Error al llamar a la API:", error);
            removeTypingIndicator(); // Quitar "escribiendo..." si hay error
            addMessage("Lo siento, tuve un problema para conectarme. Intenta de nuevo.", "bot");
        }
    };

    // --- Funciones de ayuda para el chat ---

    // Añade un mensaje (de usuario o bot) a la ventana
    function addMessage(message, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        
        const textElement = document.createElement("p");
        textElement.textContent = message;
        
        // Si es el indicador "...", le damos un id
        if (sender === "bot-typing") {
            textElement.id = "typing-indicator";
        }

        messageElement.appendChild(textElement);
        chatWindow.appendChild(messageElement);
        
        // Auto-scroll al final
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Quita el indicador "..."
    function removeTypingIndicator() {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) {
            indicator.parentElement.remove();
        }
    }

    // --- Event Listeners (para enviar) ---
    
    // 1. Enviar con el botón
    sendButton.addEventListener("click", sendMessage);
    
    // 2. Enviar con la tecla "Enter"
    chatInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    // --- Lógica de Modo Offline (Requisito PGI-007) ---
    function updateOnlineStatus() {
        if (navigator.onLine) {
            statusLight.classList.remove("offline");
            statusLight.classList.add("online");
            statusLight.title = "Conectado";
        } else {
            statusLight.classList.remove("online");
            statusLight.classList.add("offline");
            statusLight.title = "Modo Offline";
        }
    }

    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);

    // Comprobar estado al cargar la página
    updateOnlineStatus();
});
