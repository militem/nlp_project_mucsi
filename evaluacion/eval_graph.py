"""
Evaluación Topológica del Grafo de Hoteles en ArcadeDB
=======================================================
Calcula métricas estructurales del grafo sin necesidad de ground truth:
- Conteos de nodos/aristas
- Densidad
- Grado medio y distribución
- Cobertura (% hoteles con chunks, reviews, amenities, etc.)
- Componentes conexos (aproximación)

Uso:
    python eval_graph.py                          # BD por defecto
    python eval_graph.py --db hotels_rvs_srvcs_graph_llm
"""

import argparse
import json
import os
import sys
import time
import requests
from collections import Counter

# Configuración ──────────────────────────────────────────────────────────────

ARCADEDB_URL = "http://localhost:2480"
ARCADEDB_USER = "root"
ARCADEDB_PASSWORD = "hotel_nlp_2024"
DEFAULT_DB = "hotels_rvs_srvcs_graph"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

VERTEX_TYPES = ["Hotel", "Chunk", "Review", "Amenity", "Location", "City", "Source", "PriceRange"]
EDGE_TYPES = ["HAS_CHUNK", "HAS_REVIEW", "LOCATED_IN", "HAS_AMENITY",
              "NEAR_LOCATION", "FROM_SOURCE", "IN_PRICE_RANGE"]


# ArcadeDB helpers ───────────────────────────────────────────────────────────

def arcadedb_query(db_name: str, sql: str, params: dict = None) -> dict:
    payload = {"language": "sql", "command": sql}
    if params:
        payload["params"] = params
    resp = requests.post(
        f"{ARCADEDB_URL}/api/v1/command/{db_name}",
        json=payload,
        auth=(ARCADEDB_USER, ARCADEDB_PASSWORD),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def count_type(db: str, vtype: str) -> int:
    r = arcadedb_query(db, f"SELECT count(*) FROM {vtype}")
    return r["result"][0]["count(*)"]


# Métricas ───────────────────────────────────────────────────────────────────

def compute_counts(db: str) -> dict:
    """Conteos de nodos y aristas por tipo."""
    vertices = {}
    for vt in VERTEX_TYPES:
        try:
            vertices[vt] = count_type(db, vt)
        except Exception:
            vertices[vt] = 0

    edges = {}
    for et in EDGE_TYPES:
        try:
            edges[et] = count_type(db, et)
        except Exception:
            edges[et] = 0

    return {"vertices": vertices, "edges": edges,
            "total_vertices": sum(vertices.values()),
            "total_edges": sum(edges.values())}


def compute_density(counts: dict) -> float:
    """Densidad del grafo: |E| / (|V| * (|V| - 1))."""
    v = counts["total_vertices"]
    e = counts["total_edges"]
    if v <= 1:
        return 0.0
    return e / (v * (v - 1))


def compute_hotel_degrees(db: str) -> dict:
    """Grado (in + out) de cada hotel y distribución."""
    r = arcadedb_query(db,
        "SELECT hotel_id, name, "
        "out('HAS_CHUNK').size() as chunks, "
        "out('HAS_REVIEW').size() as reviews, "
        "out('HAS_AMENITY').size() as amenities, "
        "out('NEAR_LOCATION').size() as locations, "
        "out('LOCATED_IN').size() as city, "
        "out('FROM_SOURCE').size() as sources, "
        "out('IN_PRICE_RANGE').size() as price_range "
        "FROM Hotel"
    )
    hotels = r.get("result", [])

    degrees = []
    for h in hotels:
        total = (h.get("chunks", 0) + h.get("reviews", 0) +
                 h.get("amenities", 0) + h.get("locations", 0) +
                 h.get("city", 0) + h.get("sources", 0) +
                 h.get("price_range", 0))
        degrees.append({
            "hotel_id": h["hotel_id"],
            "name": h.get("name", ""),
            "degree_total": total,
            "chunks": h.get("chunks", 0),
            "reviews": h.get("reviews", 0),
            "amenities": h.get("amenities", 0),
            "locations": h.get("locations", 0),
        })

    degree_values = [d["degree_total"] for d in degrees]
    stats = {}
    if degree_values:
        sorted_d = sorted(degree_values)
        stats = {
            "min": sorted_d[0],
            "max": sorted_d[-1],
            "mean": sum(sorted_d) / len(sorted_d),
            "median": sorted_d[len(sorted_d) // 2],
        }
        # Histograma de rangos
        bins = Counter()
        for d in degree_values:
            if d <= 5:
                bins["0-5"] += 1
            elif d <= 10:
                bins["6-10"] += 1
            elif d <= 20:
                bins["11-20"] += 1
            elif d <= 50:
                bins["21-50"] += 1
            else:
                bins["50+"] += 1
        stats["histogram"] = dict(bins)

    return {"hotels": degrees, "degree_stats": stats}


def compute_coverage(db: str) -> dict:
    """Porcentaje de hoteles con al menos 1 relación de cada tipo."""
    total = count_type(db, "Hotel")
    if total == 0:
        return {}

    coverage = {}
    checks = [
        ("with_chunks", "out('HAS_CHUNK').size() > 0"),
        ("with_reviews", "out('HAS_REVIEW').size() > 0"),
        ("with_amenities", "out('HAS_AMENITY').size() > 0"),
        ("with_locations", "out('NEAR_LOCATION').size() > 0"),
        ("with_city", "out('LOCATED_IN').size() > 0"),
        ("with_source", "out('FROM_SOURCE').size() > 0"),
        ("with_price_range", "out('IN_PRICE_RANGE').size() > 0"),
    ]
    for label, cond in checks:
        try:
            r = arcadedb_query(db,
                f"SELECT count(*) FROM Hotel WHERE {cond}")
            count = r["result"][0]["count(*)"]
        except Exception:
            count = 0
        coverage[label] = {
            "count": count,
            "total": total,
            "percentage": round(count / total * 100, 1),
        }

    return coverage


def compute_top_amenities(db: str, top_n: int = 15) -> list:
    """Amenities con más conexiones entrantes (hubs de servicio)."""
    try:
        r = arcadedb_query(db,
            "SELECT name, in('HAS_AMENITY').size() as connections "
            "FROM Amenity ORDER BY connections DESC LIMIT " + str(top_n))
        return [{"name": a["name"], "connections": a["connections"]}
                for a in r.get("result", [])]
    except Exception:
        return []


def compute_reviews_stats(db: str) -> dict:
    """Estadísticas de distribución de reviews por hotel."""
    try:
        r = arcadedb_query(db,
            "SELECT hotel_id, out('HAS_REVIEW').size() as n_reviews FROM Hotel")
        counts = [h.get("n_reviews", 0) for h in r.get("result", [])]
        if not counts:
            return {}
        sorted_c = sorted(counts)
        return {
            "min": sorted_c[0],
            "max": sorted_c[-1],
            "mean": round(sum(sorted_c) / len(sorted_c), 1),
            "median": sorted_c[len(sorted_c) // 2],
            "hotels_without_reviews": sum(1 for c in counts if c == 0),
            "hotels_with_reviews": sum(1 for c in counts if c > 0),
        }
    except Exception:
        return {}


# Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación topológica del grafo")
    parser.add_argument("--db", default=DEFAULT_DB, help="Nombre de la BD en ArcadeDB")
    args = parser.parse_args()
    db = args.db

    print("=" * 70)
    print(f"  EVALUACIÓN TOPOLÓGICA DEL GRAFO – {db}")
    print("=" * 70)

    # 1. Conteos
    print("\n1. Conteos de nodos y aristas...")
    counts = compute_counts(db)
    print(f"   Vértices totales: {counts['total_vertices']}")
    for k, v in counts["vertices"].items():
        print(f"     {k:<15} {v}")
    print(f"   Aristas totales:  {counts['total_edges']}")
    for k, v in counts["edges"].items():
        print(f"     {k:<18} {v}")

    # 2. Densidad
    density = compute_density(counts)
    print(f"\n2. Densidad del grafo: {density:.6f}")

    # 3. Grados
    print("\n3. Distribución de grado de hoteles...")
    degrees = compute_hotel_degrees(db)
    stats = degrees["degree_stats"]
    if stats:
        print(f"   Min: {stats['min']}  Max: {stats['max']}  "
              f"Media: {stats['mean']:.1f}  Mediana: {stats['median']}")
        print(f"   Histograma: {stats.get('histogram', {})}")

    # 4. Cobertura
    print("\n4. Cobertura de hoteles...")
    coverage = compute_coverage(db)
    for label, data in coverage.items():
        print(f"   {label:<20} {data['count']}/{data['total']} ({data['percentage']}%)")

    # 5. Top Amenities
    print("\n5. Top amenities (por conexiones)...")
    top_am = compute_top_amenities(db)
    for i, a in enumerate(top_am[:10]):
        print(f"   {i+1:2}. {a['name']:<25} {a['connections']} hoteles")

    # 6. Reviews stats
    print("\n6. Estadísticas de reviews...")
    rv_stats = compute_reviews_stats(db)
    if rv_stats:
        for k, v in rv_stats.items():
            print(f"   {k:<30} {v}")

    # Guardar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "database": db,
        "counts": counts,
        "density": density,
        "degree_stats": degrees["degree_stats"],
        "coverage": coverage,
        "top_amenities": top_am,
        "reviews_stats": rv_stats,
    }
    out_file = os.path.join(RESULTS_DIR, f"graph_metrics_{db}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte guardado en: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
