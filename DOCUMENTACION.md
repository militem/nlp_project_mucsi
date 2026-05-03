# Documentación del Proyecto: Buscador de Hoteles mediante Lenguaje Natural

## Índice

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Estado Actual del Proyecto](#2-estado-actual-del-proyecto)
3. [Corpus Recopilado](#3-corpus-recopilado)
4. [Pipeline de Desarrollo](#4-pipeline-de-desarrollo)
   - 4.1 [Paso 1: Recopilación del Corpus (Completado)](#41-paso-1-recopilación-del-corpus-completado)
   - 4.2 [Paso 2: Análisis Exploratorio de Datos (EDA)](#42-paso-2-análisis-exploratorio-de-datos-eda)
   - 4.3 [Paso 3: Preprocesamiento de Texto](#43-paso-3-preprocesamiento-de-texto)
   - 4.4 [Paso 4: Identificación de Sesgos](#44-paso-4-identificación-de-sesgos)
   - 4.5 [Paso 5: Métricas de Evaluación](#45-paso-5-métricas-de-evaluación)
5. [Construcción del Grafo de Conocimiento](#5-construcción-del-grafo-de-conocimiento)
6. [GraphRAG: Recuperación Aumentada por Grafos](#6-graphrag-recuperación-aumentada-por-grafos)
7. [Estructura del Proyecto](#7-estructura-del-proyecto)
8. [Tecnologías y Dependencias](#8-tecnologías-y-dependencias)
9. [Resultados Esperados](#9-resultados-esperados)

---

## 1. Descripción del Proyecto

**Nombre:** Buscador de Hoteles mediante Lenguaje Natural (HotelNLP)

**Objetivo Principal:** Desarrollar un buscador de hoteles avanzado utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Minería de Texto para superar las limitaciones de las búsquedas tradicionales basadas en palabras clave, permitiendo interacciones más naturales y semánticas con la información hotelera.

### Características Clave

- **Análisis Semántico:** Comprender la intención del usuario y procesar consultas en lenguaje natural
- **Extracción de Entidades:** Identificación automática de elementos clave (ubicación, servicios, descripciones de alojamiento)
- **Procesamiento de Datos:** Extracción y limpieza de información de diversas fuentes web
- **Relaciones Semánticas:** Análisis de relaciones entre diferentes entidades
- **Sistema de Recomendación Basado en Grafos:** Grafo de conocimiento para capturar relaciones complejas
- **GraphRAG:** Generación Aumentada por Recuperación sobre grafos para respuestas contextuales

---

## 2. Estado Actual del Proyecto

| Fase | Estado | Progreso |
|------|--------|----------|
| Recopilación del Corpus | ✅ Completada | 100% |
| Análisis Exploratorio (EDA) | ⏳ Pendiente | 0% |
| Preprocesamiento | ⏳ Pendiente | 0% |
| Identificación de Sesgos | ⏳ Pendiente | 0% |
| Métricas de Evaluación | ⏳ Pendiente | 0% |
| Construcción del Grafo | ⏳ Pendiente | 0% |
| Implementación GraphRAG | ⏳ Pendiente | 0% |

---

## 3. Corpus Recopilado

### 3.1 Descripción General

El corpus consiste en datos de hoteles de alojamiento en español recopilados de **3 ciudades principales de España**:

- **Barcelona**
- **Madrid**
- **Bilbao**

### 3.2 Fuentes de Datos

| Fuente | Hoteles | Ciudades | Formato |
|--------|---------|----------|---------|
| Expedia | ~300 | Madrid, Bilbao | JSON, CSV |
| Hotels.com | ~300 | Barcelona, Bilbao, Madrid | JSON, CSV |
| Booking.com | ~300 | General | JSON |

**Total aproximado:** 550+ registros de hoteles

### 3.3 Esquema de Datos

```json
{
    "name": "Nombre del hotel",
    "address": "Dirección completa",
    "rating": "Puntuación de guests (ej. 8.8, 9.2)",
    "price": "Precio por noche en EUR",
    "services": ["lista de servicios/amenidades"],
    "url": "URL de reserva del hotel",
    "city": "Nombre de la ciudad"
}
```

### 3.4 Estadísticas del Corpus

- **Volumen total:** ~2.8 MB
- **Número de archivos:** 16
- **Idioma:** Español (servicios y direcciones)
- **Servicios por hotel:** 50+ amenidades incluyendo WiFi, desayuno, restaurantes, accesibilidad, prácticas de sostenibilidad

---

## 4. Pipeline de Desarrollo

### 4.1 Paso 1: Recopilación del Corpus (Completado) ✅

#### Objetivos Alcanzados

- [x] Implementación de web scraping para Expedia (`webscraping.py`)
- [x] Implementación de API para Booking.com (`Api.py`)
- [x] Integración de datos de Hotels.com
- [x] Conversión JSON a CSV con deduplicación
- [x] Normalización de esquemas de datos

#### Scripts Implementados

**webscraping.py (419 líneas)**
- `build_search_url()`: Construye URLs de búsqueda de Expedia con parámetros
- `setup_driver()`: Configura undetected ChromeDriver para evasión de bots
- `dismiss_popups()`: Maneja banners de cookies y popups
- `extract_hotel_urls_from_search()`: Paginación a través de resultados
- `extract_hotel_details()`: Extrae nombre, dirección, rating, precio, servicios
- `fetch_urls()` (Paso 1): Recopila URLs de hoteles
- `fetch_details()` (Paso 2): Visita cada URL y extrae detalles

**Api.py (116 líneas)**
- `get_dest_id()`: Obtiene ID de destino para una ciudad
- `get_hotel_details()`: Obtiene información detallada
- `load_existing_names()`: Evita duplicados
- `save_data()`: Guarda en booking_results.json

#### Uso

```bash
# Paso 1: Recopilar URLs
python webscraping.py --step 1 --limit 5 --city Madrid --headed

# Paso 2: Extraer detalles de hoteles
python webscraping.py --step 2 --limit 100 --batch_size 2 --city Madrid --headed

# API de Booking.com
python Api.py -c M -n 100  # Madrid
python Api.py -c B -n 100  # Barcelona
python Api.py -c P -n 100  # Bilbao
```

---

### 4.2 Paso 2: Análisis Exploratorio de Datos (EDA) ⏳

#### Objetivos

Realizar un análisis exhaustivo del corpus recopilado para comprender:

- [ ] Distribución de precios por ciudad y rating
- [ ] Distribución de ratings de hoteles
- [ ] Frecuencia de servicios/amenidades
- [ ] Longitud de descripciones de servicios
- [ ] Calidad de datos (valores faltantes, duplicados, inconsistencias)
- [ ] Análisis de sentimiento en servicios descritos
- [ ] Nube de palabras de servicios más comunes

#### Métricas a Calcular

| Métrica | Descripción |
|---------|-------------|
| `precio_promedio` | Precio medio por ciudad |
| `rating_medio` | Rating medio por ciudad |
| `servicios_unicos` | Número de servicios distintos |
| `hoteles_por_ciudad` | Distribución geográfica |
| `servicios_mas_frecuentes` | Top 20 amenidades |
| `valoraciones_faltantes` | % de ratings missing |
| `precios_faltantes` | % de precios missing |

#### Estructura Sugerida

```
eda/
├── eda_general.py           # Análisis general del corpus
├── eda_precios.py           # Análisis de precios
├── eda_servicios.py         # Análisis de servicios
├── eda_visualizaciones.py   # Gráficos y visualizaciones
└── eda_calidad.py           # Calidad de datos
```

---

### 4.3 Paso 3: Preprocesamiento de Texto ⏳

#### Objetivos

Preparar el texto para análisis NLP posterior:

- [ ] Tokenización del texto en español
- [ ] Normalización de texto (minúsculas, eliminación de tildes opcional)
- [ ] Eliminación de stopwords en español
- [ ] Lematización o Stemming
- [ ] Extracción de entidades (NER) parahoteles, ubicaciones, servicios
- [ ] Extracción de features de texto (TF-IDF, embeddings)
- [ ] Limpieza de datos inconsistentes

#### Pipeline de Preprocesamiento

```
Raw Text → Tokenización → Normalización → Stopwords → Lematización → Features
```

#### Técnicas a Utilizar

| Técnica | Herramienta | Propósito |
|---------|-------------|-----------|
| Tokenización | spaCy / NLTK | Dividir texto en tokens |
| Stopwords | spaCy (es) | Eliminar palabras comunes |
| Lematización | spaCy (es) | Reducir a forma base |
| NER | spaCy (es) | Extraer entidades |
| Embeddings | sentence-transformers | Representación semántica |

#### Estructura Sugerida

```
preprocessing/
├── text_cleaning.py        # Limpieza básica de texto
├── tokenization.py          # Tokenización
├── lemmatization.py         # Lematización
├── entity_extraction.py     # Extracción de entidades
├── feature_extraction.py    # TF-IDF y embeddings
└── preprocessing_pipeline.py # Pipeline completo
```

---

### 4.4 Paso 4: Identificación de Sesgos ⏳

#### Objetivos

Detectar y documentar sesgos potenciales en el corpus:

- [ ] Sesgo de selección (fuentes web)
- [ ] Sesgo geográfico (solo 3 ciudades)
- [ ] Sesgo de precio (gama de precios representada)
- [ ] Sesgo de idioma (solo español)
- [ ] Sesgo temporal (fecha de scrapeo)
- [ ] Sesgo de completitud (hoteles con/sin ciertos servicios)

#### Tipos de Sesgo a Analizar

| Tipo de Sesgo | Descripción | Impacto |
|---------------|-------------|---------|
| **Selección** | Hoteles solo de Booking, Expedia, Hotels.com | Puede omitir hoteles boutique o independientes |
| **Geográfico** | Solo 3 ciudades españolas | No representa otras regiones |
| **Precio** | Sesgo hacia hoteles de gama media-alta | Sous-representación de económicos/lujo |
| **Temporal** | Datos de un momento específico | Precios y disponibilidad cambiantes |
| **Idioma** | Solo español | Limitación para usuarios internacionales |
| **Completitud** | Hoteles con información incompleta | Afecta calidad de búsqueda |

#### Métricas de Sesgo

```python
# Ejemplo de métricas a calcular
bias_metrics = {
    "coverage_per_city": {},      # % de hotels por ciudad vs población
    "price_distribution": {},      # Distribución vs mercado real
    "service_completeness": {},    # % campos llenos por hotel
    "temporal_freshness": {},      # Antigüedad de datos
    "source_representation": {}     # Distribución por fuente
}
```

---

### 4.5 Paso 5: Métricas de Evaluación ⏳

#### Objetivos

Definir métricas para evaluar el sistema de búsqueda:

- [ ] Métricas de Retrieval (Precision, Recall, F1)
- [ ] Métricas de Ranking (NDCG, MRR)
- [ ] Métricas de Embeddings (Similitud coseno)
- [ ] Métricas de Extracción de Entidades (NER)
- [ ] Benchmarks de calidad de respuestas GraphRAG

#### Métricas de Retrieval

| Métrica | Fórmula | Descripción |
|---------|---------|-------------|
| Precision@K | P@K = TP/(TP+FP) | Proporción de resultados relevantes en top K |
| Recall@K | R@K = TP/(TP+FN) | Proporción de relevantes recuperados |
| F1@K | F1 = 2·P·R/(P+R) | Media armónica de P y R |
| MRR | 1/rank | Media del recíproco del rank |

#### Métricas de Ranking

| Métrica | Descripción |
|---------|-------------|
| **NDCG@K** | Normalized Discounted Cumulative Gain |
| **MAP** | Mean Average Precision |
| **Hit Rate@K** | Proporción de queries con al menos 1 acierto |

#### Pipeline de Evaluación

```
Query → Sistema → Resultados → Métricas → Reporte
         ↓
    Ground Truth → Comparación
```

---

## 5. Construcción del Grafo de Conocimiento

### 5.1 Descripción General

Construcción de un **Knowledge Graph** para representar relaciones entre hoteles, servicios, ubicaciones y atributos.

### 5.2 Arquitectura del Grafo

#### Nodos (Entities)

| Tipo de Nodo | Ejemplos |
|--------------|----------|
| `Hotel` | H10 Catalunya Plaza, Hotel Ritz |
| `Ciudad` | Barcelona, Madrid, Bilbao |
| `Servicio` | WiFi, Desayuno, Parking |
| `Categoría` | Lujo, Económico, Boutique |
| `Dirección` | Calle Gran Vía 23 |

#### Relaciones (Edges)

| Relación | Descripción | Propiedades |
|----------|-------------|-------------|
| `UBICADO_EN` | Hotel → Ciudad | - |
| `TIENE_SERVICIO` | Hotel → Servicio | rating_servicio |
| `TIENE_DIRECCION` | Hotel → Dirección | - |
| `OFRECE_PRECIO` | Hotel → Precio | moneda, fecha |
| `TIENE_RATING` | Hotel → Rating | fuente |
| `ES_CATEGORIA` | Hotel → Categoría | - |

### 5.3 Modelo de Datos (Neo4j)

```cypher
// Crear nodo Hotel
CREATE (h:Hotel {
    id: "hotel_001",
    name: "H10 Catalunya Plaza",
    rating: 8.8,
    price: 197
})

// Crear nodo Ciudad
CREATE (c:Ciudad {name: "Barcelona"})

// Crear relación
CREATE (h)-[:UBICADO_EN]->(c)

// Crear nodo Servicio
CREATE (s:Servicio {name: "WiFi", category: "tecnologia"})
CREATE (h)-[:TIENE_SERVICIO {rating: 4.5}]->(s)
```

### 5.4 Implementación

```python
# Estructura sugerida
knowledge_graph/
├── graph_construction.py     # Construcción del grafo
├── node_extraction.py        # Extracción de nodos
├── relation_extraction.py    # Extracción de relaciones
├── neo4j_integration.py      # Integración con Neo4j
└── graph_validation.py       # Validación del grafo
```

### 5.5 Pasos de Construcción

1. **Extracción de Entidades:** Identificar hoteles, ciudades, servicios
2. **Normalización:** Unificar formatos y nombres
3. **Extracción de Relaciones:** Identificar conexiones entre entidades
4. **Enriquecimiento:** Añadir propiedades y metadatos
5. **Validación:** Verificar integridad del grafo
6. **Importación:** Cargar en Neo4j

---

## 6. GraphRAG: Recuperación Aumentada por Grafos

### 6.1 Descripción General

GraphRAG combina la recuperación basada en grafos con la generación de lenguaje natural para proporcionar respuestas contextuales y precisas a consultas de usuario.

### 6.2 Arquitectura GraphRAG

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Consulta   │────▶│   Retrieval │────▶│  Generator  │
│   Usuario    │     │   (Grafo)   │     │    (LLM)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Knowledge  │
                    │    Graph    │
                    │   (Neo4j)   │
                    └─────────────┘
```

### 6.3 Componentes

| Componente | Función |
|------------|---------|
| **Query Processor** | Procesa y parsea la consulta del usuario |
| **Graph Retriever** | Recupera nodos y relaciones relevantes |
| **Vector Store** | Almacena embeddings para similitud semántica |
| **Context Builder** | Construye contexto con información del grafo |
| **Response Generator** | Genera respuesta usando LLM |

### 6.4 Flujo de Ejecución

```
1. Usuario ingresa consulta natural
       ↓
2. Query Processor analiza la consulta
       ↓
3. Graph Retriever busca en el grafo:
   - Nodos coincidentes
   - Relaciones relevantes
   - Subgrafos conectados
       ↓
4. Vector Store busca similitud semántica
       ↓
5. Context Builder fusiona resultados
       ↓
6. Response Generator produce respuesta
       ↓
7. Usuario recibe respuesta contextual
```

### 6.5 Tipos de Consultas Soportadas

| Tipo | Ejemplo | Estrategia |
|------|---------|------------|
| **Búsqueda simple** | "Hoteles con WiFi en Barcelona" | Filtrado directo |
| **Comparación** | "¿Qué hotel tiene mejor rating?" | Análisis de relaciones |
| **Recomendación** | "Hotel romántico para fin de semana" | Similitud + contexto |
| **Agregación** | "Promedio de precios por ciudad" | Consultas Cypher |
| **Razonamiento** | "Hoteles con piscina y parking" | Traversals de grafo |

### 6.6 Implementación

```python
# Estructura sugerida
graphrag/
├── query_processor.py        # Procesamiento de consultas
├── graph_retriever.py        # Recuperación del grafo
├── vector_store.py           # Almacenamiento de embeddings
├── context_builder.py        # Construcción de contexto
├── response_generator.py     # Generación de respuestas
├── evaluation.py             # Evaluación del sistema
└── api.py                    # API del servicio
```

### 6.7 Consultas Cypher de Ejemplo

```cypher
// Hoteles con WiFi gratis en Barcelona
MATCH (h:Hotel)-[:TIENE_SERVICIO]->(s:Servicio {name: "WiFi"})
MATCH (h)-[:UBICADO_EN]->(c:Ciudad {name: "Barcelona"})
RETURN h.name, h.rating, h.price

// Hoteles mejor valorados con piscina
MATCH (h:Hotel)-[:TIENE_SERVICIO]->(s:Servicio {name: "Piscina"})
WHERE h.rating >= 8.0
RETURN h.name, h.rating
ORDER BY h.rating DESC
LIMIT 10

// Servicios más comunes por ciudad
MATCH (h:Hotel)-[:UBICADO_EN]->(c:Ciudad)
MATCH (h)-[:TIENE_SERVICIO]->(s:Servicio)
RETURN c.name, s.name, COUNT(*) as count
ORDER BY count DESC
```

---

## 7. Estructura del Proyecto

```
nlp_project_mucsi/
├── README.md                    # Documentación general
├── DOCUMENTACION.md             # Este documento
│
├── webscraping.py               # Scraper de Expedia (✓ Completado)
├── Api.py                       # API de Booking.com (✓ Completado)
├── webscraping/                 # (para scrapers adicionales)
│
├── data/                        # Datos recopilados
│   ├── expedia_hotels.json      # Hoteles Expedia
│   ├── expedia_hotels.csv
│   ├── expedia_urls_*.json
│   ├── hotelscom_hotels_*_clean.json
│   ├── hotelscom_hotels_*_clean.csv
│   ├── hotelscom_urls_*.json
│   ├── booking_results.json
│   └── convert_json_to_csv.py
│
├── eda/                         # Análisis Exploratorio (⏳ Pendiente)
│   ├── eda_general.py
│   ├── eda_precios.py
│   ├── eda_servicios.py
│   ├── eda_visualizaciones.py
│   └── eda_calidad.py
│
├── preprocessing/               # Preprocesamiento (⏳ Pendiente)
│   ├── text_cleaning.py
│   ├── tokenization.py
│   ├── lemmatization.py
│   ├── entity_extraction.py
│   ├── feature_extraction.py
│   └── preprocessing_pipeline.py
│
├── bias_analysis/               # Análisis de Sesgos (⏳ Pendiente)
│   ├── selection_bias.py
│   ├── geographic_bias.py
│   ├── price_bias.py
│   └── bias_report.py
│
├── evaluation/                  # Métricas de Evaluación (⏳ Pendiente)
│   ├── retrieval_metrics.py
│   ├── ranking_metrics.py
│   ├── embedding_metrics.py
│   └── evaluation_pipeline.py
│
├── knowledge_graph/             # Grafo de Conocimiento (⏳ Pendiente)
│   ├── graph_construction.py
│   ├── node_extraction.py
│   ├── relation_extraction.py
│   ├── neo4j_integration.py
│   └── graph_validation.py
│
└── graphrag/                    # Sistema GraphRAG (⏳ Pendiente)
    ├── query_processor.py
    ├── graph_retriever.py
    ├── vector_store.py
    ├── context_builder.py
    ├── response_generator.py
    ├── evaluation.py
    └── api.py
```

---

## 8. Tecnologías y Dependencias

### 8.1 Tecnologías Core

| Tecnología | Uso | Estado |
|------------|-----|--------|
| Python 3.8+ | Lenguaje principal | ✅ |
| Pandas | Manipulación de datos | ✅ |
| spaCy | NLP en español | ⏳ |
| sentence-transformers | Embeddings | ⏳ |
| Neo4j | Grafo de conocimiento | ⏳ |
| Chromadb/Qdrant | Vector store | ⏳ |

### 8.2 Web Scraping

| Librería | Uso |
|----------|-----|
| undetected_chromedriver | Scraping de Expedia |
| Selenium | Automatización del navegador |
| BeautifulSoup | Parseo de HTML |
| requests | Llamadas HTTP |

### 8.3 Gráficos y Visualización

| Librería | Uso |
|----------|-----|
| matplotlib | Gráficos básicos |
| seaborn | Visualizaciones estadísticas |
| plotly | Gráficos interactivos |

### 8.4 Instalación de Dependencias

```bash
# Dependencias básicas
pip install pandas numpy

# Web Scraping
pip install undetected-chromedriver selenium beautifulsoup4 requests

# NLP
pip install spacy
python -m spacy download es_core_news_sm

# Embeddings
pip install sentence-transformers

# Grafo de conocimiento
pip install neo4j

# Vector store
pip install chromadb

# Visualización
pip install matplotlib seaborn plotly

# LLM
pip install openai anthropic
```

---

## 9. Resultados Esperados

### 9.1 Corpus Final

- ✅ ~550+ hoteles de 3 ciudades españolas
- [ ] Limpieza y normalización completa
- [ ] Metadatos enriquecidos
- [ ] Documentación de calidad

### 9.2 Análisis

- [ ] Reporte EDA completo con visualizaciones
- [ ] Análisis de sesgos documentado
- [ ] Preprocesamiento optimizado para español

### 9.3 Sistema GraphRAG

- [ ] Grafo de conocimiento funcional en Neo4j
- [ ] Motor de búsqueda semántica
- [ ] API de consultas en lenguaje natural
- [ ] Respuestas contextuales y precisas

### 9.4 Métricas de Éxito

| Métrica | Objetivo |
|---------|----------|
| Precision@10 | > 0.80 |
| Recall@10 | > 0.75 |
| NDCG@10 | > 0.70 |
| MRR | > 0.75 |
| Tiempo de respuesta | < 2s |

---

## Historial de Versiones

| Versión | Fecha | Descripción |
|---------|-------|-------------|
| 1.0 | - | Estado inicial: Corpus recopilado |

---

*Documento generado para el proyecto HotelNLP - Buscador de Hoteles mediante Lenguaje Natural*
