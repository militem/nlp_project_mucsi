#!/usr/bin/env python3
"""
Script para unificar los datasets de hoteles y exportar en formato JSONL
listo para anotar en Doccano.

Archivos compatibles (misma estructura):
  - booking_results.json                  → fuente: booking
  - hotelscom_hotels_Bilbao_clean.json    → fuente: hotelscom
  - hotelscom_hotels_Madrid_clean.json    → fuente: hotelscom
  - hotelscom_hotels_Barcelona_clean.json → fuente: hotelscom
  - expedia_hotels.json                   → fuente: expedia (si existe)

Transformaciones aplicadas:
  - Campo 'services' → 'text': si es array se convierte a párrafo único.
  - Campo 'rating': se normaliza a float.
  - Campo 'price': se normaliza a float; None si no es numérico.
  - Se añade 'source' para identificar el dataset de origen.
  - Se eliminan duplicados exactos (mismo name + city + url).

Salida (JSONL, un registro por línea):
  - hotels_exploration.jsonl  → 20 % (exploración / desarrollo)
  - hotels_evaluation.jsonl   → 10 % (evaluación / test)
  - hotels_corpus.jsonl       → 70 % (corpus de anotación final)
"""

import json
import random
from pathlib import Path

# ── Rutas de los archivos ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

FILES = [
    {
        "path": BASE_DIR / "booking_results.json",
        "source": "booking",
    },
    {
        "path": BASE_DIR / "hotelscom_hotels_Bilbao_clean.json",
        "source": "hotelscom",
    },
    {
        "path": BASE_DIR / "hotelscom_hotels_Madrid_clean.json",
        "source": "hotelscom",
    },
    {
        "path": BASE_DIR / "hotelscom_hotels_Barcelona_clean.json",
        "source": "hotelscom",
    },
    {
        "path": BASE_DIR / "expedia_hotels.json",
        "source": "expedia"
    }
]

OUT_EXPLORATION = BASE_DIR / "hotels_bilbao_exploration.jsonl"
OUT_EVALUATION  = BASE_DIR / "hotels_bilbao_evaluation.jsonl"
OUT_CORPUS      = BASE_DIR / "hotels_bilbao_corpus.jsonl"

RANDOM_SEED = 42  # Reproducibilidad

# ── Funciones auxiliares ───────────────────────────────────────────────────────

def normalize_rating(value):
    """Convierte el rating a float. Acepta '8,5' o '8.5' o 8.5."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return None


def normalize_price(value):
    """Convierte el precio a float. Devuelve None si no es numérico."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return None


def services_to_string(services):
    """
    Convierte el campo 'services' a un único párrafo.
    Si ya es string, lo devuelve tal cual.
    Si es lista, une los elementos con ', '.
    Si está vacío/None, devuelve cadena vacía.
    """
    if not services:
        return ""
    if isinstance(services, str):
        return services.strip()
    if isinstance(services, list):
        return ", ".join(str(s).strip() for s in services if s)
    return str(services)


def verify_schema(record):
    """Comprueba que el registro tiene los campos mínimos esperados."""
    required = {"name", "city", "url"}
    return required.issubset(record.keys())


# ── Lógica principal ───────────────────────────────────────────────────────────

def main():
    all_records = []
    seen_keys = set()  # Para deduplicar

    for file_info in FILES:
        path = file_info["path"]
        source = file_info["source"]

        if not path.exists():
            print(f"[ADVERTENCIA] Archivo no encontrado: {path}")
            continue

        print(f"\n Leyendo: {path.name}  (fuente={source})")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  [ERROR] El archivo no es un array JSON; se omite.")
            continue

        # Analizar campos del primer elemento para informar al usuario
        if data and isinstance(data[0], dict):
            sample_keys = set(data[0].keys())
            print(f"  Campos detectados: {sorted(sample_keys)}")
            expected = {"name", "address", "rating", "price", "services", "url", "city"}
            missing = expected - sample_keys
            if missing:
                print(f"  [AVISO] Campos que faltan en este archivo: {missing}")
        elif data and not isinstance(data[0], dict):
            print(f"  [ERROR] Los elementos no son objetos (dict); se omite.")
            continue

        added = skipped_dup = skipped_invalid = 0

        for record in data:
            if not isinstance(record, dict):
                skipped_invalid += 1
                continue

            if not verify_schema(record):
                skipped_invalid += 1
                continue

            # Clave de deduplicación
            dedup_key = (
                str(record.get("name", "")).strip().lower(),
                str(record.get("city", "")).strip().lower(),
                str(record.get("url", "")).strip(),
            )
            if dedup_key in seen_keys:
                skipped_dup += 1
                continue
            seen_keys.add(dedup_key)

            # Normalizar campos
            text = services_to_string(record.get("services"))
            if not text:
                skipped_invalid += 1
                continue

            normalized = {
                "name":  str(record.get("name", "")).strip(),
                "city":  str(record.get("city", "")).strip(),
                "text":  text,
                "label": None
            }

            all_records.append(normalized)
            added += 1

        print(f"  Registros añadidos: {added}  |  duplicados omitidos: {skipped_dup}  |  inválidos: {skipped_invalid}")

    # ── Filtro por ciudad ──────────────────────────────────────────────────────
    CITY_FILTER = "Bilbao"
    all_records = [r for r in all_records if r["city"].strip().lower() == CITY_FILTER.lower()]
    print(f"\n Filtro aplicado: solo hoteles de '{CITY_FILTER}' → {len(all_records)} registros")

    # ── Mezcla aleatoria y partición────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)

    total = len(all_records)
    n_exploration = int(total * 0.20)
    n_evaluation  = int(total * 0.10)

    split_exploration = all_records[:n_exploration]
    split_evaluation  = all_records[n_exploration : n_exploration + n_evaluation]
    split_corpus      = all_records[n_exploration + n_evaluation :]

    def write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(split_exploration, OUT_EXPLORATION)
    write_jsonl(split_evaluation,  OUT_EVALUATION)
    write_jsonl(split_corpus,      OUT_CORPUS)

    print(f"\n Archivos JSONL generados (seed={RANDOM_SEED}):")
    print(f"   Exploración  (20%): {len(split_exploration):>4} registros  →  {OUT_EXPLORATION.name}")
    print(f"   Evaluación   (10%): {len(split_evaluation):>4} registros  →  {OUT_EVALUATION.name}")
    print(f"   Corpus final (70%): {len(split_corpus):>4} registros  →  {OUT_CORPUS.name}")
    print(f"   TOTAL:              {total:>4} registros")



if __name__ == "__main__":
    main()
