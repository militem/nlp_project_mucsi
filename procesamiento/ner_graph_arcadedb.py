"""
NER + Grafo Hibrido Enriquecido en ArcadeDB
=============================================
1. Lee hotels_embeddings_reviews_services.json
2. NER con BERT multilingue o LLM de LM Studio
3. Construye un grafo hibrido enriquecido en ArcadeDB:
   - Nodos: Hotel, Chunk (vector), Review (vector), Amenity, Location,
            City, Source, PriceRange
   - Aristas: HAS_CHUNK, HAS_REVIEW, LOCATED_IN, HAS_AMENITY,
              NEAR_LOCATION, FROM_SOURCE, IN_PRICE_RANGE

Esquema del grafo:
  [PriceRange] <-IN_PRICE_RANGE- [Hotel] -HAS_AMENITY---> [Amenity]
  [Source]     <-FROM_SOURCE----- [Hotel] -LOCATED_IN----> [City]
                                  [Hotel] -HAS_CHUNK-----> [Chunk]
                                  [Hotel] -HAS_REVIEW----> [Review]
                                  [Hotel] -NEAR_LOCATION-> [Location]

Requisitos:
    pip install requests transformers torch

Uso:
    1. docker compose up -d
    2. python ner_graph_arcadedb.py --ner bert
       python ner_graph_arcadedb.py --ner llm
"""

import argparse
import json
import os
import re
import sys
import time
import requests
from typing import Optional

# Configuracion ──────────────────────────────────────────────────────────────

ARCADEDB_URL = "http://localhost:2480"
ARCADEDB_USER = "root"
ARCADEDB_PASSWORD = "hotel_nlp_2024"
DATABASE_NAME = "hotels_rvs_srvcs_graph_llm"

LM_STUDIO_URL = "http://192.168.1.151:1234"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "hotels_embeddings_reviews_services.json")

# Modo de NER: "bert" o "llm" (se puede sobreescribir con --ner)
NER_MODE = "bert"

LOG_EVERY = 25
REQUEST_DELAY = 0.3     # pausa entre operaciones ArcadeDB
LLM_DELAY = 1.5         # pausa entre llamadas al LLM (proteger GPU)
MAX_DB_RETRIES = 5
DB_RETRY_WAIT = 5
MAX_LLM_RETRIES = 3
LLM_RETRY_WAIT = 10
SAVE_EVERY = 50          # guardado parcial de progreso

# Palabras clave para extraer amenities con BERT (fallback sin LLM)
AMENITY_KEYWORDS = [
    "wifi", "piscina", "gimnasio", "spa", "parking", "aparcamiento",
    "restaurante", "bar", "desayuno", "buffet", "terraza", "jardin",
    "sauna", "jacuzzi", "minibar", "aire acondicionado", "lavanderia",
    "recepcion 24", "ascensor", "accesible", "mascotas", "pet-friendly",
    "cocina", "microondas", "nevera", "secador", "plancha", "caja fuerte",
    "television", "tv", "balcon", "vistas al mar", "playa", "centro",
    "boutique", "business", "familiar", "adults only", "todo incluido",
    "all-inclusive", "room service", "servicio habitaciones", "transfer",
]

# Rangos de precio
PRICE_RANGES = [
    ("budget",    0,    80),
    ("mid-range", 80,  150),
    ("luxury",   150,  300),
    ("premium",  300, 9999),
]

# Prompt para NER con LLM
NER_PROMPT = """Analyze this hotel information and extract structured entities.

Hotel Name: {name}
City: {city}
Address: {address}
Price: {price} EUR/night
Rating: {rating}/10
Source: {source}
Services: {services}

Return ONLY a valid JSON object with these exact keys:
- "amenities": list of specific amenities and services (e.g., "WiFi", "Piscina", "Gimnasio", "Restaurante", "Spa", "Parking", "Desayuno", "Bar", "Terraza"). Normalize: use consistent short names, no duplicates.
- "nearby_places": list of nearby landmarks, streets, or points of interest mentioned (e.g., "Sagrada Familia", "La Rambla", "Plaza Mayor"). Only real place names, not generic descriptions.
- "hotel_features": list of distinctive features or categories (e.g., "Adults Only", "Pet-friendly", "Boutique", "Business", "Beachfront", "All-inclusive", "24h Reception", "Accessible").

Rules:
- Keep names short and normalized (max 3-4 words each)
- Remove duplicates and near-duplicates
- Extract at most 15 amenities, 10 nearby_places, 8 hotel_features
- If a field is empty or has no relevant entities, use an empty list
- Return ONLY the JSON, no explanation"""


# Funciones auxiliares ────────────────────────────────────────────────────────

# Cache global para el pipeline de BERT (se carga una sola vez)
_bert_ner_pipeline = None


def get_bert_pipeline():
    """Carga el pipeline de BERT NER multilingue (lazy, una sola vez)."""
    global _bert_ner_pipeline
    if _bert_ner_pipeline is None:
        from transformers import pipeline
        print("  Cargando modelo BERT NER multilingue...")
        _bert_ner_pipeline = pipeline(
            "ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple",
        )
        print("  Modelo BERT NER cargado")
    return _bert_ner_pipeline


def get_price_range(price) -> Optional[str]:
    """Clasifica el precio en un rango."""
    if price in (None, "N/A", ""):
        return None
    try:
        if isinstance(price, str):
            price = float(price.replace(',', '').replace('€', '').strip())
        else:
            price = float(price)
    except ValueError:
        return None

    for label, lo, hi in PRICE_RANGES:
        if lo <= price < hi:
            return label
    return None


def bert_extract_entities(hotel: dict) -> dict:
    """Extrae entidades con BERT NER multilingue + keywords para amenities."""
    nlp = get_bert_pipeline()

    # Construir texto completo para NER
    services_data = hotel.get("services", [])
    if isinstance(services_data, list):
        services_text = ", ".join(services_data)
    else:
        services_text = str(services_data)

    # Incluir reviews en el texto para extraer ubicaciones mencionadas
    reviews_data = hotel.get("reviews", [])
    reviews_text = " ".join(
        r.get("comment", "") for r in reviews_data if r.get("comment")
    )

    full_text = (
        f"{hotel.get('name', '')}. {hotel.get('address', '')}. "
        f"{services_text}. {reviews_text}"
    )

    # Truncar a 512 tokens aprox (BERT limit)
    if len(full_text) > 2000:
        full_text = full_text[:2000]

    # Ejecutar NER
    try:
        ner_results = nlp(full_text)
    except Exception as e:
        print(f"\n    Error BERT NER: {e}")
        return empty_entities()

    # Clasificar entidades
    locations = set()
    for ent in ner_results:
        label = ent.get("entity_group", "")
        word = ent.get("word", "").strip().strip("#")
        if len(word) < 2 or len(word) > 50:
            continue
        if label == "LOC":
            locations.add(word)

    # Extraer amenities por keywords
    services_lower = services_text.lower()
    amenities = set()
    for kw in AMENITY_KEYWORDS:
        if kw in services_lower:
            amenities.add(kw.capitalize())

    return {
        "amenities": list(amenities),
        "nearby_places": list(locations),
        "hotel_features": [],  # BERT no extrae features, solo LOC/PER/ORG
    }


def extract_entities(hotel: dict, mode: str) -> dict:
    """Dispatcher: elige BERT o LLM segun el modo configurado."""
    if mode == "bert":
        return bert_extract_entities(hotel)
    else:
        return llm_extract_entities(hotel)


def llm_extract_entities(hotel: dict) -> dict:
    """Usa el LLM de LM Studio para extraer entidades estructuradas."""
    services_data = hotel.get("services", [])
    if isinstance(services_data, list):
        services_text = ", ".join(services_data)
    else:
        services_text = str(services_data)
        
    # Truncar servicios si es muy largo (max ~1500 chars para el LLM)
    if len(services_text) > 1500:
        services_text = services_text[:1500] + "..."

    prompt = NER_PROMPT.format(
        name=hotel.get("name", ""),
        city=hotel.get("city", ""),
        address=hotel.get("address", ""),
        price=hotel.get("price", "N/A"),
        rating=hotel.get("rating", "N/A"),
        source=hotel.get("source", ""),
        services=services_text,
    )

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    # Desactivar modo thinking para modelos qwen3-thinking
                    # (el bloque <think> consume todos los tokens y la respuesta queda vacia)
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return parse_llm_json(content)

        except requests.exceptions.HTTPError as e:
            if "crash" in str(e.response.text).lower() or e.response.status_code >= 500:
                wait = LLM_RETRY_WAIT * attempt
                print(f"\n    LLM crash (intento {attempt}/{MAX_LLM_RETRIES}), "
                      f"esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n    Error LLM HTTP {e.response.status_code}: "
                      f"{e.response.text[:200]}")
                return empty_entities()

        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            wait = LLM_RETRY_WAIT * attempt
            print(f"\n    LLM desconexion (intento {attempt}/{MAX_LLM_RETRIES}), "
                  f"esperando {wait}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"\n    Error inesperado LLM: {e}")
            return empty_entities()

    return empty_entities()


def parse_llm_json(content: str) -> dict:
    """Parsea la respuesta JSON del LLM con manejo robusto de errores."""
    # Eliminar bloques <think>...</think> de modelos thinking
    content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

    # Limpiar: extraer JSON si esta envuelto en markdown
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if json_match:
        content = json_match.group(1)
    # Tambien intentar encontrar {} directamente
    brace_match = re.search(r'\{[\s\S]*\}', content)
    if brace_match:
        content = brace_match.group(0)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return empty_entities()

    # Validar y limpiar
    result = {
        "amenities": [],
        "nearby_places": [],
        "hotel_features": [],
    }
    for key in result:
        raw = data.get(key, [])
        if isinstance(raw, list):
            # Filtrar items validos: strings no vacias, max 50 chars
            result[key] = list({
                item.strip()
                for item in raw
                if isinstance(item, str) and 1 < len(item.strip()) < 50
            })
    return result


def empty_entities() -> dict:
    return {"amenities": [], "nearby_places": [], "hotel_features": []}


# ArcadeDB REST API ──────────────────────────────────────────────────────────

def arcadedb_server(command: str) -> dict:
    resp = requests.post(
        f"{ARCADEDB_URL}/api/v1/server",
        json={"command": command},
        auth=(ARCADEDB_USER, ARCADEDB_PASSWORD),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def arcadedb_command(command: str, params: Optional[dict] = None) -> dict:
    """Ejecuta SQL en ArcadeDB con reintentos por desconexion."""
    payload = {"language": "sql", "command": command}
    if params:
        payload["params"] = params

    for attempt in range(1, MAX_DB_RETRIES + 1):
        try:
            resp = requests.post(
                f"{ARCADEDB_URL}/api/v1/command/{DATABASE_NAME}",
                json=payload,
                auth=(ARCADEDB_USER, ARCADEDB_PASSWORD),
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout):
            if attempt < MAX_DB_RETRIES:
                wait = DB_RETRY_WAIT * attempt
                print(f"\n    DB desconexion (intento {attempt}/{MAX_DB_RETRIES}), "
                      f"esperando {wait}s...")
                time.sleep(wait)
            else:
                raise


def wait_for_arcadedb(max_wait: int = 60):
    print(f"Esperando a ArcadeDB en {ARCADEDB_URL}...")
    for i in range(max_wait):
        try:
            resp = requests.get(
                f"{ARCADEDB_URL}/api/v1/ready",
                auth=(ARCADEDB_USER, ARCADEDB_PASSWORD),
                timeout=3,
            )
            if resp.status_code == 204 or resp.ok:
                print("  ArcadeDB disponible")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"  ...esperando ({i + 1}s)")
    print("No se pudo conectar a ArcadeDB")
    return False


def setup_database(vector_dim: int):
    """Crea/recrea la base de datos y el esquema enriquecido."""
    # Eliminar BD existente si hay
    try:
        arcadedb_server(f"drop database {DATABASE_NAME}")
        print(f"  BD '{DATABASE_NAME}' eliminada")
    except Exception:
        pass
    time.sleep(1)

    arcadedb_server(f"create database {DATABASE_NAME}")
    print(f"  BD '{DATABASE_NAME}' creada")

    print("  Creando esquema enriquecido...")

    # Vertices
    vertex_types = {
        "Hotel": [
            ("hotel_id", "STRING"), ("name", "STRING"), ("city", "STRING"),
            ("address", "STRING"), ("rating", "DOUBLE"), ("price", "DOUBLE"),
            ("services", "STRING"), ("url", "STRING"), ("source", "STRING"),
        ],
        "Chunk": [
            ("chunk_id", "STRING"), ("text", "STRING"), ("vector", "LIST"),
        ],
        "Review": [
            ("review_id", "STRING"), ("source", "STRING"), ("score", "DOUBLE"),
            ("comment", "STRING"), ("date", "STRING"), ("text", "STRING"), ("vector", "LIST"),
        ],
        "Amenity": [("name", "STRING")],
        "Location": [("name", "STRING")],
        "City": [("name", "STRING")],
        "Source": [("name", "STRING")],
        "PriceRange": [("label", "STRING"), ("min_price", "DOUBLE"), ("max_price", "DOUBLE")],
    }
    for vtype, props in vertex_types.items():
        arcadedb_command(f"CREATE VERTEX TYPE {vtype}")
        for pname, ptype in props:
            try:
                arcadedb_command(f"CREATE PROPERTY {vtype}.{pname} IF NOT EXISTS {ptype}")
            except Exception:
                pass

    # Aristas
    edge_types = [
        "HAS_CHUNK", "HAS_REVIEW", "LOCATED_IN", "HAS_AMENITY",
        "NEAR_LOCATION", "FROM_SOURCE", "IN_PRICE_RANGE",
    ]
    for etype in edge_types:
        arcadedb_command(f"CREATE EDGE TYPE {etype}")

    # Indices unicos
    indices = [
        "CREATE INDEX ON Hotel (hotel_id) UNIQUE",
        "CREATE INDEX ON Chunk (chunk_id) UNIQUE",
        "CREATE INDEX ON Review (review_id) UNIQUE",
        "CREATE INDEX ON Amenity (name) UNIQUE",
        "CREATE INDEX ON Location (name) UNIQUE",
        "CREATE INDEX ON City (name) UNIQUE",
        "CREATE INDEX ON Source (name) UNIQUE",
        "CREATE INDEX ON PriceRange (label) UNIQUE",
    ]
    for cmd in indices:
        try:
            arcadedb_command(cmd)
        except Exception:
            pass

    # Indice vectorial
    try:
        arcadedb_command(
            f"CREATE INDEX ON Chunk (vector) NULL_STRATEGY SKIP "
            f"HNSW (dimensions {vector_dim}, type FLOAT, m 16, efConstruction 128, efSearch 100)"
        )
        print(f"  Indice vectorial HNSW creado (dim={vector_dim})")
    except Exception as e:
        print(f"  Aviso indice vectorial: {e}")

    # Crear nodos Source predefinidos
    for src in ["booking", "expedia", "hotels_com", "hotelscom"]:
        arcadedb_command("INSERT INTO Source SET name = :name", {"name": src})

    # Crear nodos PriceRange predefinidos
    for label, lo, hi in PRICE_RANGES:
        arcadedb_command(
            "INSERT INTO PriceRange SET label = :label, min_price = :lo, max_price = :hi",
            {"label": label, "lo": float(lo), "hi": float(hi)},
        )

    print("  Esquema creado correctamente")


# Insercion de nodos ─────────────────────────────────────────────────────────

def get_or_create(vtype: str, field: str, value: str, cache: dict) -> str:
    """Busca o crea un vertice por campo unico. Devuelve RID."""
    key = (vtype, value.lower())
    if key in cache:
        return cache[key]

    try:
        result = arcadedb_command(
            f"SELECT FROM {vtype} WHERE {field}.toLowerCase() = :val LIMIT 1",
            {"val": value.lower()},
        )
        records = result.get("result", [])
        if records:
            cache[key] = records[0]["@rid"]
            return cache[key]
    except Exception:
        pass

    try:
        result = arcadedb_command(
            f"INSERT INTO {vtype} SET {field} = :val",
            {"val": value},
        )
        rid = result["result"][0]["@rid"]
        cache[key] = rid
        return rid
    except Exception:
        # Ya existe por concurrencia
        result = arcadedb_command(
            f"SELECT FROM {vtype} WHERE {field}.toLowerCase() = :val LIMIT 1",
            {"val": value.lower()},
        )
        rid = result["result"][0]["@rid"]
        cache[key] = rid
        return rid


def create_edge(edge_type: str, from_rid: str, to_rid: str):
    arcadedb_command(f"CREATE EDGE {edge_type} FROM {from_rid} TO {to_rid}")


# Main ────────────────────────────────────────────────────────────────────────

def main():
    # Argparse
    parser = argparse.ArgumentParser(description="NER + Grafo Hibrido en ArcadeDB")
    parser.add_argument(
        "--ner", choices=["bert", "llm"], default=NER_MODE,
        help="Modo de NER: 'bert' (BERT multilingue) o 'llm' (LM Studio)"
    )
    args = parser.parse_args()
    ner_mode = args.ner

    print("=" * 70)
    print(f"  NER ({ner_mode.upper()}) + GRAFO HIBRIDO ENRIQUECIDO")
    print("=" * 70)

    # 1. Cargar datos
    print(f"\nCargando datos desde: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: archivo no encontrado: {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        hotels = json.load(f)
    print(f"  {len(hotels)} hoteles cargados")

    # Dimension del vector
    vector_dim = 0
    for h in hotels:
        for c in h.get("chunks", []):
            if c.get("vector"):
                vector_dim = len(c["vector"])
                break
        if vector_dim > 0:
            break
    print(f"  Dimension de vectores: {vector_dim}")

    # 2. Verificar LM Studio (solo si modo llm)
    if ner_mode == "llm":
        print("\nVerificando LM Studio...")
        try:
            resp = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
            models = [m["id"] for m in resp.json()["data"]]
            chat_models = [m for m in models if "embedding" not in m.lower()]
            print(f"  Modelos disponibles: {models}")
            if chat_models:
                print(f"  Modelo de chat: {chat_models[0]}")
            else:
                print("  AVISO: no se encontro modelo de chat, NER fallara")
        except Exception as e:
            print(f"  Error conectando a LM Studio: {e}")
            sys.exit(1)
    else:
        print(f"\nModo NER: BERT multilingue (no requiere LM Studio para NER)")

    # 3. Conectar a ArcadeDB y crear esquema
    if not wait_for_arcadedb():
        print("Ejecuta primero: docker compose up -d")
        sys.exit(1)

    setup_database(vector_dim)

    # 4. Procesar hoteles
    print(f"\nProcesando hoteles con NER ({ner_mode})...")
    total = len(hotels)
    stats = {
        "chunks": 0, "reviews": 0, "amenities": 0, "locations": 0,
        "features": 0, "ner_errors": 0,
    }
    cache = {}  # cache unificado de RIDs

    # Guardar progreso de entidades NER por hotel (para reanudacion)
    ner_results_file = os.path.join(SCRIPT_DIR, f"ner_results_cache_{ner_mode}.json")
    ner_cache = {}
    if os.path.exists(ner_results_file):
        with open(ner_results_file, "r", encoding="utf-8") as f:
            ner_cache = json.load(f)
        print(f"  Cache NER cargado: {len(ner_cache)} hoteles previos")

    for idx, hotel in enumerate(hotels):
        hotel_name = hotel.get("name", "?")
        hotel_id = hotel["hotel_id"]

        # --- Insertar Hotel ---
        # Convertir datos numericos
        try:
            parsed_rating = float(hotel.get("rating")) if hotel.get("rating") not in (None, "N/A", "") else None
        except (ValueError, TypeError):
            parsed_rating = None
            
        try:
            raw_price = hotel.get("price")
            if raw_price in (None, "N/A", ""):
                parsed_price = None
            elif isinstance(raw_price, str):
                parsed_price = float(raw_price.replace(',', '').replace('€', '').strip())
            else:
                parsed_price = float(raw_price)
        except (ValueError, TypeError):
            parsed_price = None

        try:
            result = arcadedb_command(
                "INSERT INTO Hotel SET hotel_id = :hotel_id, name = :name, "
                "city = :city, address = :address, rating = :rating, "
                "price = :price, services = :services, url = :url, source = :source",
                params={
                    "hotel_id": hotel_id, "name": hotel["name"],
                    "city": hotel.get("city", ""), "address": hotel.get("address", ""),
                    "rating": parsed_rating, "price": parsed_price,
                    "services": json.dumps(hotel.get("services", [])),
                    "url": hotel.get("url", ""), "source": hotel.get("source", ""),
                },
            )
            hotel_rid = result["result"][0]["@rid"]
        except Exception as e:
            print(f"\n  Error insertando hotel '{hotel_name}': {e}")
            continue

        # --- Arista Hotel -> City ---
        city_name = hotel.get("city", "")
        if city_name:
            try:
                city_rid = get_or_create("City", "name", city_name, cache)
                create_edge("LOCATED_IN", hotel_rid, city_rid)
            except Exception:
                pass

        # --- Arista Hotel -> Source (de reviews) ---
        review_sources = set()
        for rev in hotel.get("reviews", []):
            src = rev.get("source", "")
            if src:
                review_sources.add(src)
        for src_name in review_sources:
            try:
                src_rid = get_or_create("Source", "name", src_name, cache)
                create_edge("FROM_SOURCE", hotel_rid, src_rid)
            except Exception:
                pass

        # --- Arista Hotel -> PriceRange ---
        price_label = get_price_range(hotel.get("price"))
        if price_label:
            try:
                pr_result = arcadedb_command(
                    "SELECT FROM PriceRange WHERE label = :label LIMIT 1",
                    {"label": price_label},
                )
                if pr_result.get("result"):
                    create_edge("IN_PRICE_RANGE", hotel_rid,
                                pr_result["result"][0]["@rid"])
            except Exception:
                pass

        # --- Insertar Chunks de SERVICIOS (solo type=service) ---
        for chunk in hotel.get("chunks", []):
            if chunk.get("type") != "service":
                continue
            try:
                result = arcadedb_command(
                    "INSERT INTO Chunk SET chunk_id = :chunk_id, "
                    "text = :text, vector = :vector",
                    params={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk.get("text", ""),
                        "vector": chunk.get("vector", []),
                    },
                )
                chunk_rid = result["result"][0]["@rid"]
                create_edge("HAS_CHUNK", hotel_rid, chunk_rid)
                stats["chunks"] += 1
            except Exception as e:
                print(f"\n    Error chunk {chunk.get('chunk_id')}: {e}")
            time.sleep(REQUEST_DELAY)

        # --- Insertar Reviews como vertices Review con vectores ---
        for chunk in hotel.get("chunks", []):
            if chunk.get("type") != "review":
                continue
            try:
                score = chunk.get("score")
                if score == "N/A" or score is None:
                    score = None
                else:
                    score = float(score)

                # Buscar la review original para obtener source y date
                review_source = ""
                review_date = ""
                review_comment = chunk.get("text", "")
                rev_idx_str = chunk.get("chunk_id", "").split("_review_")[-1]
                try:
                    rev_idx = int(rev_idx_str)
                    reviews_list = hotel.get("reviews", [])
                    if rev_idx < len(reviews_list):
                        review_source = reviews_list[rev_idx].get("source", "")
                        review_date = reviews_list[rev_idx].get("date", "")
                        review_comment = reviews_list[rev_idx].get("comment", review_comment)
                except (ValueError, IndexError):
                    pass

                result = arcadedb_command(
                    "INSERT INTO Review SET review_id = :review_id, "
                    "source = :source, score = :score, comment = :comment, "
                    "date = :date, text = :text, vector = :vector",
                    params={
                        "review_id": chunk["chunk_id"],
                        "source": review_source,
                        "score": score,
                        "comment": review_comment,
                        "date": review_date,
                        "text": chunk.get("text", ""),
                        "vector": chunk.get("vector", []),
                    },
                )
                review_rid = result["result"][0]["@rid"]
                create_edge("HAS_REVIEW", hotel_rid, review_rid)
                stats["reviews"] += 1
            except Exception as e:
                print(f"\n    Error review {chunk.get('chunk_id')}: {e}")
            time.sleep(REQUEST_DELAY)

        # --- NER ---
        if hotel_id in ner_cache:
            entities = ner_cache[hotel_id]
        else:
            entities = extract_entities(hotel, ner_mode)
            ner_cache[hotel_id] = entities
            if ner_mode == "llm":
                time.sleep(LLM_DELAY)

        if not any(entities.values()):
            stats["ner_errors"] += 1

        # --- Aristas Hotel -> Amenity ---
        for amenity_name in entities.get("amenities", []):
            try:
                amenity_rid = get_or_create("Amenity", "name", amenity_name, cache)
                create_edge("HAS_AMENITY", hotel_rid, amenity_rid)
                stats["amenities"] += 1
            except Exception:
                pass

        # --- Aristas Hotel -> Location ---
        for loc_name in entities.get("nearby_places", []):
            try:
                loc_rid = get_or_create("Location", "name", loc_name, cache)
                create_edge("NEAR_LOCATION", hotel_rid, loc_rid)
                stats["locations"] += 1
            except Exception:
                pass

        # --- Hotel features como Amenity (tipo especial) ---
        for feat_name in entities.get("hotel_features", []):
            try:
                feat_rid = get_or_create("Amenity", "name", feat_name, cache)
                create_edge("HAS_AMENITY", hotel_rid, feat_rid)
                stats["features"] += 1
            except Exception:
                pass

        # Progreso
        print(f"\r  [{idx + 1}/{total}] {hotel_name[:45]:<45}", end="", flush=True)
        if (idx + 1) % LOG_EVERY == 0:
            print(f" -- ch:{stats['chunks']} rv:{stats['reviews']} "
                  f"am:{stats['amenities']} loc:{stats['locations']}")

        # Guardado parcial del cache NER
        if (idx + 1) % SAVE_EVERY == 0:
            with open(ner_results_file, "w", encoding="utf-8") as f:
                json.dump(ner_cache, f, ensure_ascii=False)

        time.sleep(REQUEST_DELAY)

    # Guardado final del cache NER
    with open(ner_results_file, "w", encoding="utf-8") as f:
        json.dump(ner_cache, f, ensure_ascii=False, indent=2)

    print()

    # 5. Estadisticas finales
    try:
        counts = {}
        for vtype in ["Hotel", "Chunk", "Review", "Amenity", "Location",
                       "City", "Source", "PriceRange"]:
            r = arcadedb_command(f"SELECT count(*) FROM {vtype}")
            counts[vtype] = r["result"][0]["count(*)"]
        edge_counts = {}
        for etype in ["HAS_CHUNK", "HAS_REVIEW", "LOCATED_IN", "HAS_AMENITY",
                       "NEAR_LOCATION", "FROM_SOURCE", "IN_PRICE_RANGE"]:
            r = arcadedb_command(f"SELECT count(*) FROM {etype}")
            edge_counts[etype] = r["result"][0]["count(*)"]
    except Exception:
        counts = {}
        edge_counts = {}

    print("\n" + "=" * 70)
    print("  PROCESO COMPLETADO")
    print("=" * 70)
    print(f"  NER modo:          {ner_mode}")
    print("  NODOS:")
    for k, v in counts.items():
        print(f"    {k:<15} {v}")
    print("  ARISTAS:")
    for k, v in edge_counts.items():
        print(f"    {k:<18} {v}")
    print(f"  Errores NER:       {stats['ner_errors']}")
    print(f"  ArcadeDB:          {ARCADEDB_URL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
