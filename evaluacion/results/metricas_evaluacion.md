# Métricas de Evaluación del Sistema Hotel GraphRAG

El sistema de recomendación de hoteles se evalúa en dos etapas secuenciales para garantizar un análisis riguroso y modular. La primera fase mide la efectividad de la recuperación de información pura (**Retrieval Metrics**), mientras que la segunda fase evalúa la calidad del texto final generado por el Modelo de Lenguaje (**Generative RAG Metrics**).

---

## 1. Métricas de Recuperación (Retrieval)
Estas métricas miden la calidad, velocidad y dispersión geométrica de los fragmentos de información (*chunks*, reseñas u hoteles) que los motores (Vectorial, Grafo, GNN) logran extraer de la base de datos antes de enviárselos al LLM. 

*   **Coherencia (Relevancia Semántica):** 
    *   **Definición:** Calcula la similitud coseno promedio entre el vector (embedding) de la pregunta del usuario y los vectores de los documentos recuperados.
    *   **Objetivo:** Medir qué tan alineada semánticamente está la información recuperada con la intención de búsqueda (valores cercanos a 1 indican alta relevancia directa).
*   **Diversidad Temática:**
    *   **Definición:** Distancia coseno promedio entre los propios documentos recuperados.
    *   **Objetivo:** Garantizar que no se le entregue al LLM información redundante. Un valor más alto indica que el contexto recuperado aporta información variada y complementaria.
*   **Número de Hoteles Únicos:**
    *   **Definición:** Cantidad de hoteles distintos presentes en el top-K de resultados recuperados.
    *   **Objetivo:** Medir la capacidad del sistema de proveer un abanico de opciones para que el usuario pueda elegir, en lugar de recuperar 5 reseñas que hablen del mismo hotel.
*   **Latencia (ms):**
    *   **Definición:** Tiempo exacto transcurrido durante el proceso de recuperación, medido en milisegundos.
    *   **Objetivo:** Evaluar la viabilidad del algoritmo para su despliegue en producción e interacciones en tiempo real.

---

## 2. Métricas Generativas (RAG y LLM-as-a-Judge)
Una vez que el sistema RAG recibe los documentos, el LLM genera una respuesta en lenguaje natural. Para evaluar esta respuesta, se ha implementado el paradigma **LLM-as-a-Judge** (inspirado en el framework RAGAS), donde un LLM evaluador (juez) califica la respuesta final en dimensiones específicas usando una escala del **1 al 5**.

*   **Faithfulness (Fidelidad o Precisión factual):**
    *   **Definición:** ¿La respuesta generada se basa *exclusivamente* en el contexto proporcionado? 
    *   **Interpretación:** Un 5 significa ausencia total de alucinaciones (no se inventan servicios que el hotel no tiene en la base de datos). Un 1 implica una alucinación severa o invención de datos.
*   **Answer Relevancy (Relevancia de la Respuesta):**
    *   **Definición:** ¿La respuesta aborda directamente la pregunta del usuario con recomendaciones concretas?
    *   **Interpretación:** Un 5 indica recomendaciones específicas con justificaciones. Un 1 penaliza respuestas evasivas ("No encontré hoteles").
*   **Context Relevancy (Relevancia del Contexto):**
    *   **Definición:** ¿El contexto inyectado al modelo era útil para responder a la pregunta?
    *   **Interpretación:** Un 5 significa que todos los resultados recuperados contenían datos críticos para resolver la duda. Permite detectar si el LLM tuvo que esforzarse para extraer valor de fragmentos irrelevantes (Retrieval pobre).
*   **Completeness (Completitud):**
    *   **Definición:** ¿La respuesta cubre *todos* los aspectos y restricciones que el usuario impuso en su pregunta?
    *   **Interpretación:** Un 5 indica que se han resuelto todas las dudas. Un valor bajo puede darse si, ante la pregunta "Hotel en el centro con parking", el LLM ignora el criterio del parking.
*   **Hit Rate / Answer Found (Tasa de Éxito):**
    *   **Definición:** Métrica booleana (True/False) agregada luego como porcentaje.
    *   **Interpretación:** Responde a la pregunta: *¿Logró el sistema recomendar al menos un hotel al final de su pipeline?* Penaliza rotundamente las respuestas del tipo "Basado en la información provista, no hay hoteles". Sirve para comprobar la robustez y capacidad de hallazgo global de cada sistema.
