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

### 4.1 Paso 1: Recopilación del Corpus (Completado) 

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

### 4.2 Paso 2: Análisis Exploratorio de Datos (EDA) 

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

### 4.3 Paso 3: Preprocesamiento de Texto 

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
| NER | BERT y LLM | Extraer entidades |
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

### 4.4 Paso 4: Identificación de Sesgos 

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

### 4.5 Paso 5: Métricas de Evaluación

> **Nota:** No se dispone de ground truth etiquetado. Todas las métricas están diseñadas para funcionar sin labels humanos, utilizando métricas topológicas, coherencia semántica y LLM-as-Judge.

#### Objetivos

- [x] Métricas topológicas del grafo (densidad, cobertura, conectividad)
- [x] Comparación NER: BERT multilingüe vs LLM (yield, Jaccard, consistencia)
- [x] Métricas de Retrieval sin ground truth (coherencia, diversidad, latencia)
- [x] Comparación RAG Vectorial vs GraphRAG Híbrido (LLM-as-Judge)

#### 4.5.1 Métricas del Grafo (`eval_graph.py`)

| Métrica | Descripción |
|---------|-------------|
| **Densidad** | `\|E\| / (\|V\| × (\|V\|-1))` — ratio aristas/aristas posibles |
| **Grado medio** | Media de conexiones por Hotel |
| **Distribución de grado** | Histograma de conexiones (detectar hubs y aislados) |
| **Cobertura** | % hoteles con ≥1 chunk, review, amenity, location, source |
| **Top Amenities** | Amenities con más conexiones entrantes (hubs) |
| **Reviews stats** | Min, max, media, mediana de reviews por hotel |

#### 4.5.2 Comparación NER (`eval_ner_comparison.py`)

| Métrica | Descripción |
|---------|-------------|
| **Yield** | Media de entidades extraídas por hotel por categoría |
| **Cobertura** | % hoteles con ≥1 amenity / location / feature |
| **Jaccard** | Solapamiento entre entidades BERT y LLM por hotel |
| **Diversidad léxica** | Vocabulario único global por categoría |
| **Consistencia** | % de keywords en servicios detectadas por NER |

#### 4.5.3 Métricas de Retrieval sin Ground Truth (`eval_retrieval.py`)

| Métrica | Descripción |
|---------|-------------|
| **Coherencia Semántica** | Similitud coseno media entre query y chunks devueltos |
| **Diversidad (ILS)** | 1 − similitud media entre resultados (Intra-List Similarity) |
| **Cobertura de Hoteles** | Nº de hoteles únicos en top-K resultados |
| **Latencia** | Tiempo de respuesta por estrategia (ms) |

Se evalúan 3 estrategias: Vectorial puro, Grafo puro e Híbrido.

#### 4.5.4 Comparación RAG vs GraphRAG (`eval_rag_comparison.py`)

Evaluación con **LLM-as-Judge** (sin ground truth):

| Métrica | Escala | Descripción |
|---------|--------|-------------|
| **Faithfulness** | 1-5 | ¿La respuesta se basa solo en el contexto dado? |
| **Answer Relevancy** | 1-5 | ¿La respuesta contesta la pregunta del usuario? |
| **Context Relevancy** | 1-5 | ¿Los chunks recuperados son relevantes? |
| **Completeness** | 1-5 | ¿La respuesta cubre todos los aspectos? |

#### Pipeline de Evaluación

```
Query → Embedding → [Vectorial / Grafo / Híbrido] → Contexto → LLM → Respuesta
                                                                          ↓
                                                              LLM-as-Judge → Scores
```

---

## 5. Construcción del Grafo de Conocimiento

### 5.1 Descripción General

Construcción de un **Knowledge Graph híbrido** en **ArcadeDB** que combina datos estructurados (propiedades de hoteles, precios, ratings) con datos no estructurados (embeddings de servicios y reviews) para permitir búsqueda semántica y por relaciones.

### 5.2 Arquitectura del Grafo (ArcadeDB)

#### Nodos (Vertices)

| Tipo de Nodo | Propiedades | Ejemplos |
|--------------|-------------|----------|
| `Hotel` | hotel_id, name, city, address, rating, price, services, url, source | H10 Catalunya Plaza |
| `Chunk` | chunk_id, text, vector (embedding) | Descripción de servicios segmentada |
| `Review` | review_id, source, score, comment, date, text, vector (embedding) | "8/10 Muy bueno, ubicación excelente" |
| `Amenity` | name | WiFi, Piscina, Spa, Parking |
| `Location` | name | La Rambla, Placa Catalunya |
| `City` | name | Barcelona, Madrid, Bilbao |
| `Source` | name | booking, expedia, hotels_com |
| `PriceRange` | label, min_price, max_price | budget (0-80€), mid (80-150€) |

#### Relaciones (Edges)

| Relación | Dirección | Descripción |
|----------|-----------|-------------|
| `HAS_CHUNK` | Hotel → Chunk | Conecta hotel con sus chunks de servicios + embedding |
| `HAS_REVIEW` | Hotel → Review | Conecta hotel con sus reviews + embedding |
| `HAS_AMENITY` | Hotel → Amenity | Amenidades extraídas por NER |
| `NEAR_LOCATION` | Hotel → Location | Ubicaciones cercanas extraídas por NER |
| `LOCATED_IN` | Hotel → City | Ciudad del hotel |
| `FROM_SOURCE` | Hotel → Source | Fuentes de datos (reviews) |
| `IN_PRICE_RANGE` | Hotel → PriceRange | Clasificación por rango de precio |

### 5.3 Esquema del Grafo

```
[PriceRange] ←IN_PRICE_RANGE— [Hotel] —HAS_AMENITY→ [Amenity]
[Source]     ←FROM_SOURCE————— [Hotel] —LOCATED_IN——→ [City]
                                [Hotel] —HAS_CHUNK———→ [Chunk]  (+ vector)
                                [Hotel] —HAS_REVIEW——→ [Review] (+ vector)
                                [Hotel] —NEAR_LOCATION→ [Location]
```

### 5.4 Implementación

```python
# Pipeline implementado
procesamiento/
├── chunking_embedding.py       # Chunking + embeddings (services + reviews)
├── ner_graph_arcadedb.py       # NER (BERT/LLM) + construcción del grafo
├── docker-compose.yml          # ArcadeDB en Docker
└── ner_results_cache_*.json    # Cache de entidades NER
```

### 5.5 NER: Extracción de Entidades

Se implementaron dos métodos de NER:

| Método | Modelo | Extrae | Ventaja |
|--------|--------|--------|---------|
| **BERT** | `Davlan/bert-base-multilingual-cased-ner-hrl` | LOC + keywords amenities | Rápido, sin dependencia externa |
| **LLM** | qwen3-4b (LM Studio) | amenities, nearby_places, hotel_features | Más rico en contexto |

```bash
# Ejecución
python ner_graph_arcadedb.py --ner bert   # BERT multilingüe
python ner_graph_arcadedb.py --ner llm    # LLM (requiere LM Studio)
```

---

## 6. GraphRAG: Recuperación Aumentada por Grafos

### 6.1 Descripción General

GraphRAG combina la recuperación basada en grafos con la generación de lenguaje natural para proporcionar respuestas contextuales y precisas a consultas de usuario.

### 6.2 Arquitectura GraphRAG

```
┌─────────────┐     ┌──────────────────────────┐     ┌─────────────┐
│   Consulta   │────▶│   Retrieval Híbrido      │────▶│  Generator  │
│   Usuario    │     │ Vector + Grafo ArcadeDB  │     │ (LLM gemma-3-4b) │
└─────────────┘     └──────────────────────────┘     └─────────────┘
                           │           │
                           ▼           ▼
                    ┌──────────┐ ┌──────────┐
                    │ Embeddings│ │ Knowledge│
                    │ (Chunks/ │ │  Graph   │
                    │  Reviews)│ │(ArcadeDB)│
                    └──────────┘ └──────────┘
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
# Pipeline implementado
evaluacion/
├── eval_graph.py              # Métricas topológicas del grafo
├── eval_ner_comparison.py     # Comparación BERT vs LLM NER
├── eval_retrieval.py          # Evaluación vectorial/grafo/híbrido
├── eval_rag_comparison.py     # RAG Vectorial vs GraphRAG + LLM-as-Judge
├── queries_test.json          # 10 queries de test
└── results/                   # Reportes JSON generados
```

### 6.7 Consultas SQL (ArcadeDB)

```sql
-- Hoteles con WiFi en Barcelona
SELECT h.name, h.rating, h.price
FROM Hotel h
WHERE h.city = 'Barcelona'
AND h.hotel_id IN (
  SELECT in.hotel_id FROM HAS_AMENITY
  WHERE out.name = 'Wifi'
)

-- Top hoteles por rating con piscina
SELECT h.name, h.rating,
       out('HAS_AMENITY').name as amenities
FROM Hotel h
WHERE out('HAS_AMENITY').name CONTAINS 'Piscina'
ORDER BY h.rating DESC LIMIT 10

-- Reviews de un hotel con sus scores
SELECT r.comment, r.score, r.source
FROM Review r
WHERE r.review_id IN (
  SELECT out.review_id FROM HAS_REVIEW
  WHERE in.hotel_id = 'hotel_0001'
)
```

---

## 7. Estructura del Proyecto

```
proyecto/
├── nlp_project_mucsi/
│   ├── README.md                    # Documentación general
│   ├── DOCUMENTACION.md             # Este documento
│   ├── webscraping.py               # Scraper de Expedia (✅)
│   ├── Api.py                       # API de Booking.com (✅)
│   └── data/
│       ├── merged_hotels.json       # Dataset consolidado (680 hoteles)
│       ├── expedia_hotels.json
│       ├── booking_results.json
│       └── hotelscom_hotels_*.json
│
├── procesamiento/                   # Pipeline NLP (✅)
│   ├── chunking_embedding.py        # Chunking + embeddings
│   ├── ner_graph_arcadedb.py        # NER + grafo ArcadeDB
│   ├── docker-compose.yml           # ArcadeDB en Docker
│   ├── hotels_embeddings_reviews_services.json
│   └── ner_results_cache_*.json     # Cache NER (bert/llm)
│
└── evaluacion/                      # Evaluación (✅)
    ├── eval_graph.py                # Métricas topológicas
    ├── eval_ner_comparison.py       # BERT vs LLM NER
    ├── eval_retrieval.py            # Retrieval vectorial/grafo/híbrido
    ├── eval_rag_comparison.py       # RAG vs GraphRAG + LLM-as-Judge
    ├── queries_test.json            # 10 queries de test
    └── results/                     # Reportes generados
```

---

## 8. Tecnologías y Dependencias

### 8.1 Tecnologías Core

| Tecnología | Uso | Estado |
|------------|-----|--------|
| Python 3.10+ | Lenguaje principal | ✅ |
| ArcadeDB | Grafo de conocimiento + vector store | ✅ |
| sentence-transformers | Embeddings multilingües | ✅ |
| HuggingFace Transformers | BERT NER multilingüe | ✅ |
| LM Studio (qwen3-4b) | NER con LLM + generación RAG | ✅ |
| Docker | Contenedor ArcadeDB | ✅ |

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

# NER y LLM
pip install transformers torch requests
```

---

## 9. Resultados Esperados

### 9.1 Corpus Final

- ✅ 680 hoteles de 3 ciudades españolas (Barcelona, Madrid, Bilbao)
- ✅ 3881 reviews integradas
- ✅ Datos consolidados en `merged_hotels.json`
- ✅ Embeddings generados (servicios + reviews)

### 9.2 Grafo de Conocimiento

- ✅ Grafo híbrido en ArcadeDB con 8 tipos de vértices
- ✅ NER dual (BERT multilingüe + LLM)
- ✅ Vectores embebidos en el grafo (Chunk + Review)
- ✅ 7 tipos de aristas para navegación relacional

### 9.3 Sistema de Evaluación

- ✅ Métricas topológicas del grafo
- ✅ Comparación NER (BERT vs LLM)
- ✅ Evaluación de retrieval (vectorial/grafo/híbrido)
- ✅ Comparación RAG vs GraphRAG con LLM-as-Judge

### 9.4 Métricas de Éxito

| Métrica | Objetivo |
|---------|----------|
| Coherencia semántica media | > 0.60 |
| Diversidad ILS | > 0.30 |
| Faithfulness (LLM-judge) | > 3.5/5 |
| Answer Relevancy (LLM-judge) | > 3.5/5 |
| Cobertura NER (BERT) | > 90% hoteles |
| Latencia retrieval | < 2s |

---

## Historial de Versiones

| Versión | Fecha | Descripción |
|---------|-------|-------------|
| 1.0 | - | Estado inicial: Corpus recopilado |

---

*Documento generado para el proyecto HotelNLP - Buscador de Hoteles mediante Lenguaje Natural*
