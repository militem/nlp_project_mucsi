# Análisis Comparativo de Estrategias de Recuperación y RAG (Hotel GraphRAG)

El presente análisis evalúa el rendimiento de cuatro estrategias de recuperación (Vectorial, Grafo puro, Híbrido/GraphRAG y Redes Neuronales en Grafos - GNN) aplicadas al sistema de recomendación de hoteles. La evaluación se divide en dos fases: rendimiento de la recuperación (Retrieval) y evaluación generativa usando el paradigma LLM-as-a-Judge.

## 1. Evaluación de Retrieval (Recuperación)

La métrica de coherencia semántica (similitud coseno promedio) y la diversidad temática de los documentos recuperados arrojan los siguientes resultados agregados sobre el grafo enriquecido (`hotels_rvs_srvcs_graph`):

| Estrategia | Coherencia | Diversidad | Nro. Hoteles | Latencia (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Vectorial** | 0.646 | 0.202 | 4.7 | ~2,458 |
| **Grafo puro** | 0.000 | 1.000 | 0.0 | ~140 |
| **Híbrido** | 0.646 | 0.197 | 5.0 | ~2,589 |
| **GNN** | 0.023 | 0.045 | 5.0 | ~37 |

### Observaciones de Retrieval:
1. **El cuello de botella Vectorial**: Las aproximaciones Vectorial e Híbrida presentan latencias muy elevadas (~2.5 segundos) debido al escaneo lineal y cálculo de similitud coseno iterativo en Python contra toda la base de datos de text chunks/reseñas.
2. **Grafo Puro Inefectivo**: La búsqueda estructural estricta por sí sola falla en recuperar hoteles consistentes (0.0 resultados en promedio) dada la ambigüedad de las consultas de lenguaje natural que no matchean exactamente los nodos.
3. **Eficiencia Extrema de la GNN**: El enfoque basado en Graph Neural Networks (GraphSAGE + BPR) logra una reducción de latencia drástica, pasando de ~2.5s a apenas **37 ms** para la recuperación. Sin embargo, su coherencia textual frente a las *queries* cae significativamente (0.023), sugiriendo un desajuste temporal entre el espacio latente topológico de la GNN y el modelo base de embeddings de texto.

---

## 2. Evaluación Generativa (LLM-as-a-Judge)

Utilizando el prompt de evaluación en la fase generativa (RAG), medimos el sistema según 4 pilares en una escala de 1 a 5 y evaluamos el porcentaje de éxito resolutivo (*Hit Rate*):

| Estrategia | Faithfulness | Answer Relevancy | Context Relevancy | Completeness | Hit Rate | Latencia Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vectorial** | 4.60 | 3.80 | 4.40 | 3.00 | 70.0% | ~9-11s |
| **GraphRAG** | **5.00** | **4.00** | **4.40** | **3.20** | **80.0%** | ~10-14s |
| **GNN** | 4.70 | 3.10 | 4.00 | 2.30 | 70.0% | **~1.5-3.0s** |

### Análisis de los Pilares Generativos:

*   **GraphRAG como Estado del Arte en Precisión:** El enfoque híbrido (GraphRAG) resulta ganador en todas las métricas de calidad. Alcanza una precisión (*faithfulness*) perfecta (5.00), lo que indica que no produce alucinaciones y se ciñe totalmente a la topología y datos semánticos recuperados. Logra el mayor ratio de respuesta (*Hit Rate* del 80%).
*   **Vectorial sufre en Consultas Complejas:** Para consultas multidimensionales (ej. "terraza y restaurante en el centro"), el modelo puramente vectorial fracasa (*found=False*) porque los chunks de texto suelen estar desconectados, mientras que las aproximaciones basadas en relaciones logran juntar las piezas.
*   **GNN: Promesa de Escalabilidad pero Falta de Contexto:** Aunque la estrategia GNN logra un *Hit Rate* equivalente a la aproximación vectorial pura (70%), la completitud de las respuestas cae (2.30). Esto sucede porque la GNN recupera los hoteles correctos basándose en cercanía topológica, pero al prescindir del texto subyacente para la búsqueda, en ocasiones el contexto textual inyectado al LLM generador es insuficiente para justificar el *porqué* de la recomendación, impactando la puntuación de completitud y relevancia percibida por el evaluador.
*   **Mejora de Latencia en Pipeline RAG:** El uso de GNNs demuestra ser invaluable para el tiempo real. Completar el ciclo completo (Retrieval + Generación de LLM) toma solo ~2 segundos con GNN, contra más de 10 segundos con los métodos tradicionales y GraphRAG.

## 3. Conclusiones y Futuras Líneas (Para la Memoria)

1.  **GraphRAG es el modelo superior para calidad:** La combinación de indexación semántica (vectores) con conocimiento estructurado y relacional (Grafo) provee el contexto más robusto para los Modelos de Lenguaje, anulando las alucinaciones (*faithfulness* de 5.0).
2.  **Trade-off entre Latencia y Precisión:** El motor GNN introducido en las Fases 2 y 3 demuestra de forma empírica que se puede resolver el severo cuello de botella de la recuperación vectorial mediante una representación latente unificada, reduciendo el tiempo de recuperación en un factor de 60x.
3.  **Siguientes pasos sugeridos:** La calidad de la GNN puede equipararse a GraphRAG si se implementa un modelo *Dual Encoder*. Actualmente, la BPR Loss aleja el embedding del hotel del espacio semántico puro de las *queries*. Entrenar la red para alinear simultáneamente las consultas directas con la topología mitigaría la caída en *Context Relevancy*.
