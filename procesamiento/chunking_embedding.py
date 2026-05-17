"""
Script de Chunking + Embedding para hoteles
=============================================
- Lee hotels_unified.json
- Aplica chunking fijo de 500 caracteres con overlap de 100 al campo "services"
- Genera embeddings con un modelo de LM Studio local (API compatible con OpenAI)
- Guarda el resultado en hotels_embeddings.json con la estructura:
    {
      hotel_id, name, city, address, rating, price, services, url, source,
      chunks: [ { chunk_id, text, vector }, ... ]
    }

Requisitos:
    pip install requests

Uso:
    python chunking_embedding.py

    Asegúrate de tener LM Studio corriendo en http://localhost:1234
    con un modelo de embeddings cargado.
"""

import json
import os
import sys
import time
import requests
from typing import Optional

# Configuración ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # caracteres por chunk
CHUNK_OVERLAP = 100    # caracteres de solapamiento entre chunks

LM_STUDIO_BASE_URL = "http://192.168.1.151:1234"
LM_STUDIO_EMBEDDINGS_ENDPOINT = f"{LM_STUDIO_BASE_URL}/v1/embeddings"

# Rutas (relativas al script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # proyecto/
INPUT_FILE = os.path.join(
    PROJECT_DIR, "nlp_project_mucsi", "data", "merged_hotels.json"
)
OUTPUT_FILE = os.path.join(
    PROJECT_DIR, "procesamiento", "hotels_embeddings_reviews_services.json"
)

# Reintentos en caso de error (incluye crash del modelo)
MAX_RETRIES = 5
RETRY_DELAY = 5        # segundos entre reintentos normales
CRASH_WAIT = 30        # segundos de espera cuando el modelo crashea

# Pausa entre cada peticion de embedding para no saturar la GPU
GPU_COOLDOWN_DELAY = 1.0  # segundos entre cada embedding

# Guardado parcial cada N hoteles por si hay crash
SAVE_EVERY = 50


# Funciones ──────────────────────────────────────────────────────────────────

def fixed_chunking(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide un texto en chunks de tamaño fijo (en caracteres) con solapamiento.
    
    Args:
        text: texto a dividir.
        chunk_size: tamaño de cada chunk en caracteres.
        overlap: número de caracteres de solapamiento entre chunks consecutivos.
    
    Returns:
        Lista de strings con los chunks generados.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Avanzar: tamaño del chunk menos el overlap
        start += chunk_size - overlap

        # Si ya hemos incluido todo el texto, parar
        if end >= text_len:
            break

    return chunks


def get_embedding(text: str, model: Optional[str] = None) -> list[float]:
    """
    Obtiene el embedding de un texto usando la API de LM Studio (compatible OpenAI).
    Envia un solo texto a la vez para no saturar la GPU.
    Incluye reintentos robustos para crashes del modelo.
    
    Args:
        text: texto para generar el embedding.
        model: nombre del modelo (opcional, LM Studio usa el modelo cargado por defecto).
    
    Returns:
        Vector de embedding como lista de floats.
    """
    payload = {"input": text}
    if model:
        payload["model"] = model

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                LM_STUDIO_EMBEDDINGS_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                print(f"\n    Error de conexion (intento {attempt}/{MAX_RETRIES}). "
                      f"Reintentando en {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\nNo se pudo conectar a LM Studio en {LM_STUDIO_BASE_URL}")
                print("   Asegurate de que LM Studio este corriendo con un modelo de embeddings cargado.")
                sys.exit(1)

        except requests.exceptions.ReadTimeout:
            if attempt < MAX_RETRIES:
                wait = CRASH_WAIT * attempt
                print(f"\n    Timeout en lectura (intento {attempt}/{MAX_RETRIES}). "
                      f"Esperando {wait}s para que la GPU se recupere...")
                time.sleep(wait)
            else:
                print(f"\nTimeout repetido - la GPU no responde.")
                sys.exit(1)

        except requests.exceptions.HTTPError as e:
            resp_text = response.text
            # Detectar crash del modelo y reintentar
            if "crashed" in resp_text.lower() or response.status_code == 400:
                if attempt < MAX_RETRIES:
                    wait = CRASH_WAIT * attempt
                    print(f"\n    Modelo crasheado (intento {attempt}/{MAX_RETRIES}). "
                          f"Esperando {wait}s para que se recupere...")
                    time.sleep(wait)
                else:
                    print(f"\nEl modelo ha crasheado {MAX_RETRIES} veces. Abortando.")
                    print(f"   Respuesta: {resp_text}")
                    sys.exit(1)
            else:
                print(f"\nError HTTP de LM Studio: {e}")
                print(f"   Respuesta: {resp_text}")
                sys.exit(1)

        except (KeyError, IndexError) as e:
            print(f"\nRespuesta inesperada de LM Studio: {e}")
            print(f"   Respuesta: {response.text}")
            sys.exit(1)


def check_lm_studio_connection() -> bool:
    """Verifica que LM Studio esté corriendo y accesible."""
    try:
        response = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        print(f"LM Studio conectado. Modelos disponibles:")
        for m in models.get("data", []):
            print(f"   - {m.get('id', 'desconocido')}")
        return True
    except Exception as e:
        print(f"No se puede conectar a LM Studio: {e}")
        return False


# Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CHUNKING + EMBEDDING - Hotels Unified")
    print(f"  Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    print("=" * 70)

    # 1. Verificar conexión con LM Studio
    print("\nVerificando conexión con LM Studio...")
    if not check_lm_studio_connection():
        print("\nPara usar este script:")
        print("   1. Abre LM Studio")
        print("   2. Carga un modelo de embeddings (ej: nomic-embed-text)")
        print("   3. Activa el servidor local en el puerto 1234")
        sys.exit(1)

    # 2. Cargar datos
    print(f"\nCargando datos desde: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"Error archivo no encontrado: {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        hotels = json.load(f)

    total_hotels = len(hotels)
    print(f"   * {total_hotels} hoteles cargados")

    # 3. Procesar cada hotel: chunking + embedding (uno a uno)
    print(f"\nProcesando hoteles (embedding de 1 en 1, delay={GPU_COOLDOWN_DELAY}s)...")
    results = []
    total_chunks = 0
    skipped = 0

    for idx, hotel in enumerate(hotels):
        hotel_id = f"hotel_{idx:04d}"
        
        # Procesar services
        services_data = hotel.get("services", [])
        if isinstance(services_data, list):
            services_text = "\n".join(services_data)
        else:
            services_text = str(services_data)

        # Recopilar todos los textos a vectorizar (servicios + reviews)
        raw_chunks = []
        
        if services_text.strip():
            service_chunks_texts = fixed_chunking(services_text)
            for st in service_chunks_texts:
                raw_chunks.append({
                    "type": "service",
                    "text": st
                })
                
        # Procesar reviews
        reviews_data = hotel.get("reviews", [])
        for rev in reviews_data:
            score = rev.get("score", "N/A")
            comment = rev.get("comment", "").strip()
            if comment:
                raw_chunks.append({
                    "type": "review",
                    "score": score,
                    "text": f"Review (Nota: {score}/10): {comment}"
                })

        if not raw_chunks:
            skipped += 1
            # Incluimos el hotel pero con lista de chunks vacia
            result = {
                "hotel_id": hotel_id,
                "name": hotel.get("name", ""),
                "city": hotel.get("city", ""),
                "address": hotel.get("address", ""),
                "rating": hotel.get("rating"),
                "price": hotel.get("price"),
                "services": services_data,
                "url": hotel.get("url", ""),
                "source": hotel.get("source", ""),
                "reviews": reviews_data,
                "chunks": []
            }
            results.append(result)

            # Progreso
            print(f"\r   [{idx + 1}/{total_hotels}] {hotel.get('name', '?')[:40]:<40} "
                  f"- sin datos (skip)", end="", flush=True)
            continue

        # Generar embeddings UNO A UNO para no saturar la GPU
        chunks_data = []
        
        service_counter = 0
        review_counter = 0
        
        for raw_chunk in raw_chunks:
            embedding = get_embedding(raw_chunk["text"])
            
            if raw_chunk["type"] == "service":
                chunk_id = f"{hotel_id}_service_{service_counter:03d}"
                service_counter += 1
                chunk_record = {
                    "chunk_id": chunk_id,
                    "type": "service",
                    "text": raw_chunk["text"],
                    "vector": embedding
                }
            else:
                chunk_id = f"{hotel_id}_review_{review_counter:03d}"
                review_counter += 1
                chunk_record = {
                    "chunk_id": chunk_id,
                    "type": "review",
                    "score": raw_chunk["score"],
                    "text": raw_chunk["text"],
                    "vector": embedding
                }
                
            chunks_data.append(chunk_record)
            # Pausa entre cada embedding
            time.sleep(GPU_COOLDOWN_DELAY)

        total_chunks += len(chunks_data)

        # Construir registro completo del hotel
        result = {
            "hotel_id": hotel_id,
            "name": hotel.get("name", ""),
            "city": hotel.get("city", ""),
            "address": hotel.get("address", ""),
            "rating": hotel.get("rating"),
            "price": hotel.get("price"),
            "services": services_data,
            "url": hotel.get("url", ""),
            "source": hotel.get("source", ""),
            "reviews": reviews_data,
            "chunks": chunks_data
        }
        results.append(result)

        # Progreso
        dim = len(chunks_data[0]["vector"]) if chunks_data else 0
        print(f"\r   [{idx + 1}/{total_hotels}] {hotel.get('name', '?')[:40]:<40} "
              f"- {len(chunks_data)} chunks (dim={dim})", end="", flush=True)

        # Guardado parcial cada N hoteles
        if (idx + 1) % SAVE_EVERY == 0:
            print(f"\n   -- Guardado parcial ({idx + 1} hoteles)...")
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print()  # Salto de linea tras el progreso

    # 4. Guardar resultado
    print(f"\nGuardando resultado en: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

    # 5. Resumen
    print("\n" + "=" * 70)
    print("  PROCESO COMPLETADO")
    print("=" * 70)
    print(f"  Hoteles procesados:   {total_hotels}")
    print(f"  Hoteles con chunks:   {total_hotels - skipped}")
    print(f"  Hoteles sin services: {skipped}")
    print(f"  Total de chunks:      {total_chunks}")
    print(f"  Tamaño del archivo:   {file_size_mb:.2f} MB")
    print(f"  Archivo guardado en:  {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
