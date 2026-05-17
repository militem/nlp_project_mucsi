# Arquitectura de Procesamiento y Modelado (Pipeline RAG-GNN)

El módulo de procesamiento (`proyecto/procesamiento/`) encapsula todo el ciclo de vida de los datos, desde la limpieza del texto crudo hasta la construcción del grafo de conocimiento y el entrenamiento de la Red Neuronal de Grafos (GNN). Este flujo se ha diseñado como una secuencia modular (Pipeline) compuesta por 4 etapas principales:

## 1. Segmentación y Vectorización Textual (`chunking_embedding.py`)
Esta es la capa base del procesamiento de lenguaje natural. Se encarga de ingerir los datos estructurados provenientes del web scraping y prepararlos para la búsqueda semántica.
*   **Chunking Estratégico:** Divide las descripciones largas de los hoteles y las reseñas en fragmentos (chunks) más pequeños y digeribles, preservando el contexto semántico sin superar el límite de tokens de los modelos de embedding.
*   **Vectorización:** Cada fragmento de texto se procesa a través de un modelo de embedding (ej. `all-MiniLM-L6-v2` u otro endpoint local configurado) para convertir el texto en representaciones vectoriales densas.
*   **Persistencia:** Los resultados se almacenan en archivos intermedios como `hotels_embeddings_reviews_services.json`, sirviendo como base fundamental para la posterior ingesta de datos.

## 2. Ingesta de Grafo y Extracción NER (`ner_graph_arcadedb.py`)
Esta fase transforma el corpus de texto semi-estructurado en un Grafo de Conocimiento (Knowledge Graph) rico y explícito alojado en **ArcadeDB**.
*   **NER Dual (BERT vs LLM):** El script implementa un sistema flexible para el Reconocimiento de Entidades Nombradas. Puede extraer amenidades, servicios y localizaciones usando un enfoque tradicional de *spaCy/BERT* o aprovechar un modelo fundacional (LLM, ej. Gemma) para extracciones más ricas y complejas.
*   **Diseño Ontológico:** Construye dinámicamente la topología en ArcadeDB. Modela los datos creando nodos diferenciados (`Hotel`, `Chunk`, `Review`, `Amenity`, `Location`) y conectándolos mediante aristas explícitas (ej. `HAS_CHUNK`, `HAS_REVIEW`, `HAS_AMENITY`, `NEAR_LOCATION`).
*   **Caché:** Para no saturar la inferencia del LLM en reejecuciones, implementa un sistema de caché de resultados NER (`ner_results_cache_*.json`).

## 3. Extracción Topológica para Deep Learning (`gnn_dataset_builder.py`)
Este script actúa como el puente entre la base de datos de grafos (ArcadeDB) y el ecosistema de PyTorch. Transforma el conocimiento almacenado en estructuras tensoriales para el entrenamiento de redes neuronales.
*   **Conversión a PyTorch Geometric:** Extrae todos los nodos y enlaces bidireccionales usando consultas SQL-like de ArcadeDB, conformando un objeto `HeteroData` complejo.
*   **Alineamiento Semántico:** Se encarga de que los nodos extraídos por NER (amenidades y ubicaciones) que carecen de vectores originales sean vectorizados "al vuelo" conectándose a la API local de embeddings, logrando que el 100% de los nodos tengan un feature inicial (`embed_dim`).
*   **Mapeo:** Exporta el dataset topológico final en formato `.pt` junto con un archivo `mappings.json` fundamental para poder relacionar el espacio latente del tensor con los identificadores reales de la base de datos de ArcadeDB.

## 4. Aprendizaje de Representaciones con Grafos (`gnn_model.py`)
El núcleo avanzado del proyecto. Entrena una Red Neuronal de Grafos para calcular embeddings de hoteles que no solo dependen de texto crudo, sino de su contexto relacional completo (servicios que ofrece, reseñas asociadas, ubicación).
*   **Arquitectura Heterogénea Bidireccional:** Usa capas de paso de mensajes (`HeteroConv` envolviendo `SAGEConv`). Incorpora dinámicamente **aristas inversas** de forma automática (`Chunk -> Hotel`, `Review -> Hotel`) asegurando que los nodos "Hotel" actúen como sumideros de información y aprendan de su vecindad topológica.
*   **Warm-Start:** En lugar de inicializar los hoteles con vectores de ceros o ruido aleatorio (lo que estancaría el gradiente), los inicializa con el centroide (promedio) de los embeddings de sus propios *chunks* de texto.
*   **Aprendizaje Contrastivo (BPR Loss):** Entrena bajo un esquema *auto-supervisado* (Bayesian Personalized Ranking Loss). Optimiza el espacio latente acercando el vector del hotel a sus reseñas/chunks reales (positivos) y alejándolo de reseñas seleccionadas al azar (negativos).
*   **Exportación para Retrieval:** El producto final es un archivo `_gnn_embeddings.pt` donde cada hotel está representado por un vector enriquecido que codifica su identidad holística. Este vector reduce el tiempo de búsqueda semántica (Retrieval) en inferencia real de segundos a microsegundos (latencia de ~37ms como se comprobó en la fase de evaluación).

## Diagrama de Flujo del Pipeline
1. Datos Crudos JSON → `chunking_embedding.py` → JSON Vectorizado
2. JSON Vectorizado → `ner_graph_arcadedb.py` → Grafo en ArcadeDB
3. Grafo en ArcadeDB → `gnn_dataset_builder.py` → Tensores de PyTorch (`HeteroData.pt`)
4. PyTorch Dataset → `gnn_model.py` → Embeddings GNN Optimizados (`_gnn_embeddings.pt`)
5. **Evaluación** (`evaluacion/`) usa los Embeddings GNN para comparar contra GraphRAG y Vectorial.
