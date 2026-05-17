"""
Comparación NER: BERT multilingüe vs LLM
==========================================
Compara las entidades extraídas por ambos métodos sin ground truth:
- Yield (entidades por hotel)
- Cobertura (% hoteles con entidades)
- Solapamiento inter-método (Jaccard)
- Diversidad léxica (vocabulario único)
- Consistencia (keyword matching contra servicios)

Uso:
    python eval_ner_comparison.py
"""

import json
import os
import sys
from collections import Counter

# Configuración ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "procesamiento")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

BERT_CACHE = os.path.join(PROC_DIR, "ner_results_cache_bert.json")
LLM_CACHE = os.path.join(PROC_DIR, "ner_results_cache_llm.json")
EMBEDDINGS_FILE = os.path.join(PROC_DIR, "hotels_embeddings_reviews_services.json")


# Funciones ──────────────────────────────────────────────────────────────────

def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  AVISO: {path} no encontrado")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_yield(cache: dict) -> dict:
    """Media de entidades extraídas por hotel en cada categoría."""
    categories = ["amenities", "nearby_places", "hotel_features"]
    totals = {c: [] for c in categories}

    for hotel_id, ents in cache.items():
        for c in categories:
            totals[c].append(len(ents.get(c, [])))

    stats = {}
    for c in categories:
        values = totals[c]
        if values:
            stats[c] = {
                "mean": round(sum(values) / len(values), 2),
                "min": min(values),
                "max": max(values),
                "total": sum(values),
            }
        else:
            stats[c] = {"mean": 0, "min": 0, "max": 0, "total": 0}

    return stats


def compute_coverage(cache: dict) -> dict:
    """% de hoteles con al menos 1 entidad en cada categoría."""
    categories = ["amenities", "nearby_places", "hotel_features"]
    total = len(cache)
    if total == 0:
        return {}

    coverage = {}
    for c in categories:
        with_ents = sum(1 for ents in cache.values() if ents.get(c))
        coverage[c] = {
            "count": with_ents,
            "total": total,
            "percentage": round(with_ents / total * 100, 1),
        }

    # Hoteles sin ninguna entidad
    empty = sum(1 for ents in cache.values() if not any(ents.values()))
    coverage["completely_empty"] = {
        "count": empty,
        "total": total,
        "percentage": round(empty / total * 100, 1),
    }

    return coverage


def compute_vocabulary(cache: dict) -> dict:
    """Vocabulario único global por categoría."""
    categories = ["amenities", "nearby_places", "hotel_features"]
    vocab = {c: set() for c in categories}

    for ents in cache.values():
        for c in categories:
            for item in ents.get(c, []):
                vocab[c].add(item.lower().strip())

    return {c: {"unique_count": len(v), "top_10": sorted(v)[:10]}
            for c, v in vocab.items()}


def compute_jaccard(bert_cache: dict, llm_cache: dict) -> dict:
    """Solapamiento Jaccard entre BERT y LLM para hoteles comunes."""
    common_ids = set(bert_cache.keys()) & set(llm_cache.keys())
    if not common_ids:
        return {"common_hotels": 0}

    categories = ["amenities", "nearby_places", "hotel_features"]
    jaccards = {c: [] for c in categories}

    for hid in common_ids:
        bert_ents = bert_cache[hid]
        llm_ents = llm_cache[hid]

        for c in categories:
            set_b = {x.lower().strip() for x in bert_ents.get(c, [])}
            set_l = {x.lower().strip() for x in llm_ents.get(c, [])}

            if not set_b and not set_l:
                continue  # Ambos vacíos, no es informativo

            union = set_b | set_l
            inter = set_b & set_l
            j = len(inter) / len(union) if union else 0.0
            jaccards[c].append(j)

    stats = {"common_hotels": len(common_ids)}
    for c in categories:
        values = jaccards[c]
        if values:
            stats[c] = {
                "mean_jaccard": round(sum(values) / len(values), 3),
                "pairs_evaluated": len(values),
                "perfect_overlap": sum(1 for v in values if v == 1.0),
                "zero_overlap": sum(1 for v in values if v == 0.0),
            }
        else:
            stats[c] = {"mean_jaccard": 0, "pairs_evaluated": 0}

    return stats


def compute_consistency(cache: dict, hotels: list) -> dict:
    """Verifica si amenities conocidas en 'services' fueron detectadas por NER.

    Busca keywords como 'piscina', 'wifi', 'parking' en el texto de
    servicios y comprueba si el NER las extrajo.
    """
    keywords = [
        "wifi", "piscina", "gimnasio", "spa", "parking", "restaurante",
        "bar", "desayuno", "terraza", "sauna", "jacuzzi",
    ]

    # Indexar hoteles por hotel_id
    hotel_map = {}
    for h in hotels:
        hotel_map[h.get("hotel_id", "")] = h

    found_in_services = 0
    detected_by_ner = 0

    for hid, ents in cache.items():
        hotel = hotel_map.get(hid)
        if not hotel:
            continue

        services = hotel.get("services", [])
        if isinstance(services, list):
            services_text = " ".join(services).lower()
        else:
            services_text = str(services).lower()

        amenities_lower = {a.lower() for a in ents.get("amenities", [])}

        for kw in keywords:
            if kw in services_text:
                found_in_services += 1
                if any(kw in am for am in amenities_lower):
                    detected_by_ner += 1

    return {
        "keywords_found_in_services": found_in_services,
        "detected_by_ner": detected_by_ner,
        "consistency_rate": round(detected_by_ner / found_in_services * 100, 1)
                           if found_in_services > 0 else 0,
    }


# Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  COMPARACIÓN NER: BERT vs LLM")
    print("=" * 70)

    bert = load_cache(BERT_CACHE)
    llm = load_cache(LLM_CACHE)

    # Cargar hoteles para consistency check
    hotels = []
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            hotels = json.load(f)

    report = {}

    for label, cache in [("BERT", bert), ("LLM", llm)]:
        if not cache:
            print(f"\n  {label}: Sin datos")
            continue

        print(f"\n{'─' * 35} {label} {'─' * 35}")
        print(f"  Hoteles procesados: {len(cache)}")

        # Yield
        y = compute_yield(cache)
        print(f"\n  Yield (entidades/hotel):")
        for c, s in y.items():
            print(f"    {c:<20} media={s['mean']:>5}  "
                  f"min={s['min']}  max={s['max']}  total={s['total']}")

        # Cobertura
        cov = compute_coverage(cache)
        print(f"\n  Cobertura:")
        for c, s in cov.items():
            print(f"    {c:<20} {s['count']}/{s['total']} ({s['percentage']}%)")

        # Vocabulario
        vocab = compute_vocabulary(cache)
        print(f"\n  Vocabulario único:")
        for c, s in vocab.items():
            print(f"    {c:<20} {s['unique_count']} términos")

        # Consistency
        if hotels:
            cons = compute_consistency(cache, hotels)
            print(f"\n  Consistencia (keyword matching):")
            print(f"    Keywords en servicios: {cons['keywords_found_in_services']}")
            print(f"    Detectadas por NER:    {cons['detected_by_ner']}")
            print(f"    Tasa de consistencia:  {cons['consistency_rate']}%")
        else:
            cons = {}

        report[label] = {
            "total_hotels": len(cache),
            "yield": y,
            "coverage": cov,
            "vocabulary": vocab,
            "consistency": cons,
        }

    # Jaccard
    if bert and llm:
        print(f"\n{'─' * 30} SOLAPAMIENTO {'─' * 30}")
        jaccard = compute_jaccard(bert, llm)
        print(f"  Hoteles comunes: {jaccard['common_hotels']}")
        for c in ["amenities", "nearby_places", "hotel_features"]:
            if c in jaccard:
                j = jaccard[c]
                print(f"  {c:<20} Jaccard={j['mean_jaccard']:.3f}  "
                      f"(pares={j['pairs_evaluated']}  "
                      f"overlap_total={j['perfect_overlap']}  "
                      f"overlap_cero={j['zero_overlap']})")
        report["jaccard"] = jaccard

    # Guardar
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, "ner_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte guardado en: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
