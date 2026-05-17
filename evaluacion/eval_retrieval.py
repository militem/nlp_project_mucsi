"""
Evaluación de Retrieval: Vectorial vs Grafo vs Híbrido
========================================================
Evalúa 3 estrategias de búsqueda SIN ground truth:

1. Vectorial puro   – cosine similarity contra chunks/reviews
2. Grafo puro       – traversals SQL sobre aristas del grafo
3. Híbrido          – combinación vector + grafo con re-ranking

Métricas:
- Coherencia semántica (coseno query↔resultados)
- Diversidad ILS (Intra-List Similarity)
- Cobertura de hoteles únicos
- Latencia

Uso:
    python eval_retrieval.py
    python eval_retrieval.py --db hotels_rvs_srvcs_graph --top_k 5
"""

import argparse
import json
import math
import os
import sys
import time
import requests
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

# Configuración ──────────────────────────────────────────────────────────────

ARCADEDB_URL = "http://localhost:2480"
ARCADEDB_USER = "root"
ARCADEDB_PASSWORD = "hotel_nlp_2024"
DEFAULT_DB = "hotels_rvs_srvcs_graph"

LM_STUDIO_URL = "http://192.168.1.151:1234"    # LLM remoto (chat/NER)
EMBEDDING_URL = "http://localhost:1234"          # Embeddings local

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
QUERIES_FILE = os.path.join(SCRIPT_DIR, "queries_test.json")

TOP_K = 5


# Helpers ────────────────────────────────────────────────────────────────────

def arcadedb_query(db: str, sql: str, params: dict = None) -> dict:
    payload = {"language": "sql", "command": sql}
    if params:
        payload["params"] = params
    resp = requests.post(
        f"{ARCADEDB_URL}/api/v1/command/{db}",
        json=payload, auth=(ARCADEDB_USER, ARCADEDB_PASSWORD), timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_embedding(text: str) -> list[float]:
    """Obtiene embedding desde el modelo local."""
    resp = requests.post(
        f"{EMBEDDING_URL}/v1/embeddings",
        json={"input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def cosine_sim(a: list, b: list) -> float:
    """Similitud coseno entre dos vectores."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def flatten_vector(vec) -> list:
    """Aplana vectores de ArcadeDB que vienen como [[float, ...]]."""
    if not vec:
        return []
    if isinstance(vec[0], list):
        return vec[0]
    return vec


# Estrategias de Retrieval ──────────────────────────────────────────────────

def retrieve_vector(db: str, query_vec: list, top_k: int = TOP_K) -> list:
    """Retrieval vectorial: busca los chunks+reviews más similares."""
    # Buscar en Chunks (servicios)
    try:
        r = arcadedb_query(db,
            "SELECT chunk_id, text, vector, "
            "in('HAS_CHUNK')[0].hotel_id as hotel_id, "
            "in('HAS_CHUNK')[0].name as hotel_name "
            f"FROM Chunk LIMIT 1000")
        chunks = r.get("result", [])
    except Exception:
        chunks = []

    # Buscar en Reviews
    try:
        r = arcadedb_query(db,
            "SELECT review_id, text, vector, score, "
            "in('HAS_REVIEW')[0].hotel_id as hotel_id, "
            "in('HAS_REVIEW')[0].name as hotel_name "
            f"FROM Review LIMIT 1000")
        reviews = r.get("result", [])
    except Exception:
        reviews = []

    # Calcular similitud para todos
    scored = []
    for item in chunks:
        vec = flatten_vector(item.get("vector", []))
        if not vec:
            continue
        sim = cosine_sim(query_vec, vec)
        scored.append({
            "id": item.get("chunk_id", ""),
            "type": "service",
            "hotel_id": item.get("hotel_id", ""),
            "hotel_name": item.get("hotel_name", ""),
            "text": item.get("text", "")[:200],
            "similarity": round(sim, 4),
            "vector": vec,
        })

    for item in reviews:
        vec = flatten_vector(item.get("vector", []))
        if not vec:
            continue
        sim = cosine_sim(query_vec, vec)
        scored.append({
            "id": item.get("review_id", ""),
            "type": "review",
            "hotel_id": item.get("hotel_id", ""),
            "hotel_name": item.get("hotel_name", ""),
            "text": item.get("text", "")[:200],
            "similarity": round(sim, 4),
            "vector": vec,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def retrieve_graph(db: str, query: str, top_k: int = TOP_K) -> list:
    """Retrieval por grafo: busca hoteles por keywords en amenities/locations."""
    keywords = query.lower().split()
    stop_words = {"hotel", "con", "en", "de", "y", "el", "la", "los", "las",
                  "un", "una", "para", "por", "qué", "que", "mejor", "vs",
                  "comparar", "tiene", "tiene", "buenas", "opiniones", "sobre"}
    keywords = [k for k in keywords if k not in stop_words and len(k) > 2]

    results = []
    seen_hotels = set()

    for kw in keywords:
        # Buscar en Amenity
        try:
            r = arcadedb_query(db,
                "SELECT name, in('HAS_AMENITY') as hotels FROM Amenity "
                "WHERE name.toLowerCase() LIKE :kw",
                {"kw": f"%{kw}%"})
            for am in r.get("result", []):
                for h in am.get("hotels", []):
                    hid = h if isinstance(h, str) else str(h)
                    if hid not in seen_hotels:
                        seen_hotels.add(hid)
        except Exception:
            pass

        # Buscar en Location
        try:
            r = arcadedb_query(db,
                "SELECT name, in('NEAR_LOCATION') as hotels FROM Location "
                "WHERE name.toLowerCase() LIKE :kw",
                {"kw": f"%{kw}%"})
            for loc in r.get("result", []):
                for h in loc.get("hotels", []):
                    hid = h if isinstance(h, str) else str(h)
                    if hid not in seen_hotels:
                        seen_hotels.add(hid)
        except Exception:
            pass

        # Buscar en City
        try:
            r = arcadedb_query(db,
                "SELECT name, in('LOCATED_IN') as hotels FROM City "
                "WHERE name.toLowerCase() LIKE :kw",
                {"kw": f"%{kw}%"})
            for city in r.get("result", []):
                for h in city.get("hotels", []):
                    hid = h if isinstance(h, str) else str(h)
                    if hid not in seen_hotels:
                        seen_hotels.add(hid)
        except Exception:
            pass

    # Recuperar info de hoteles encontrados
    for rid in list(seen_hotels)[:top_k]:
        try:
            r = arcadedb_query(db,
                f"SELECT hotel_id, name, rating, price, city, "
                f"out('HAS_AMENITY').name as amenities "
                f"FROM {rid}")
            if r.get("result"):
                h = r["result"][0]
                results.append({
                    "id": rid,
                    "type": "graph",
                    "hotel_id": h.get("hotel_id", ""),
                    "hotel_name": h.get("name", ""),
                    "text": f"{h.get('name', '')} - {h.get('city', '')} - "
                            f"Rating: {h.get('rating', 'N/A')} - "
                            f"Amenities: {', '.join(h.get('amenities', [])[:5])}",
                    "similarity": 0,
                    "vector": [],
                })
        except Exception:
            pass

    return results[:top_k]


def retrieve_hybrid(db: str, query: str, query_vec: list,
                     top_k: int = TOP_K, weight_vec: float = 0.6) -> list:
    """Retrieval híbrido: combina vector + grafo con re-ranking."""
    vec_results = retrieve_vector(db, query_vec, top_k=top_k * 2)
    graph_results = retrieve_graph(db, query, top_k=top_k * 2)

    # Unificar por hotel_id
    hotel_scores = {}

    for r in vec_results:
        hid = r.get("hotel_id", "")
        if hid not in hotel_scores:
            hotel_scores[hid] = {"vector_sim": 0, "graph_hit": 0, "data": r}
        hotel_scores[hid]["vector_sim"] = max(
            hotel_scores[hid]["vector_sim"], r["similarity"])

    for r in graph_results:
        hid = r.get("hotel_id", "")
        if hid not in hotel_scores:
            hotel_scores[hid] = {"vector_sim": 0, "graph_hit": 0, "data": r}
        hotel_scores[hid]["graph_hit"] = 1.0

    # Score combinado
    results = []
    for hid, scores in hotel_scores.items():
        combined = (weight_vec * scores["vector_sim"] +
                    (1 - weight_vec) * scores["graph_hit"])
        item = scores["data"].copy()
        item["similarity"] = round(combined, 4)
        item["type"] = "hybrid"
        results.append(item)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def retrieve_gnn(db: str, query_vec: list, top_k: int = TOP_K) -> list:
    """Retrieval basado en los embeddings enriquecidos de la GNN."""
    if torch is None:
        return []
    
    data_dir = os.path.join(SCRIPT_DIR, "..", "data", "gnn")
    emb_path = os.path.join(data_dir, f"{db}_gnn_embeddings.pt")
    map_path = os.path.join(data_dir, f"{db}_mappings.json")
    
    if not os.path.exists(emb_path) or not os.path.exists(map_path):
        return []
        
    # Lazy load (en un script de prod esto iría al inicio)
    gnn_embs = torch.load(emb_path, map_location='cpu', weights_only=False) # [num_hotels, emb_dim]
    with open(map_path, "r") as f:
        mappings = json.load(f) # {idx_str: hotel_id}
        
    q_tensor = torch.tensor([query_vec], dtype=torch.float32)
    sims = F.cosine_similarity(q_tensor, gnn_embs)
    
    top_scores, top_indices = torch.topk(sims, min(top_k * 2, len(sims)))
    
    results = []
    seen = set()
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        hid = mappings.get(str(idx))
        if not hid or hid in seen: continue
        seen.add(hid)
        
        try:
            r = arcadedb_query(db,
                "SELECT hotel_id, name, city, rating, price, "
                "out('HAS_AMENITY').name as amenities "
                f"FROM Hotel WHERE hotel_id = :hid",
                {"hid": hid})
            if r.get("result"):
                h = r["result"][0]
                results.append({
                    "id": hid,
                    "type": "gnn",
                    "hotel_id": h.get("hotel_id", ""),
                    "hotel_name": h.get("name", ""),
                    "text": f"{h.get('name', '')} - {h.get('city', '')} - "
                            f"Rating: {h.get('rating', 'N/A')} - "
                            f"Amenities: {', '.join(h.get('amenities', [])[:5])}",
                    "similarity": round(score, 4),
                    "vector": gnn_embs[idx].tolist(),
                })
        except Exception:
            pass
            
        if len(results) >= top_k:
            break
            
    return results

# Métricas ───────────────────────────────────────────────────────────────────

def metric_coherence(query_vec: list, results: list) -> float:
    """Similitud coseno media entre query y resultados."""
    sims = []
    for r in results:
        vec = r.get("vector", [])
        if vec:
            sims.append(cosine_sim(query_vec, vec))
    return round(sum(sims) / len(sims), 4) if sims else 0.0


def metric_diversity_ils(results: list) -> float:
    """Intra-List Similarity: 1 - media de similitud entre resultados."""
    vecs = [r["vector"] for r in results if r.get("vector")]
    if len(vecs) < 2:
        return 1.0

    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine_sim(vecs[i], vecs[j]))

    avg_sim = sum(sims) / len(sims) if sims else 0
    return round(1 - avg_sim, 4)


def metric_hotel_coverage(results: list) -> int:
    """Número de hoteles únicos en los resultados."""
    return len({r.get("hotel_id", "") for r in results if r.get("hotel_id")})


# Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación de retrieval")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  EVALUACIÓN DE RETRIEVAL – {args.db}")
    print("=" * 70)

    # Cargar queries
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"  {len(queries)} queries de test")

    report = {"database": args.db, "top_k": args.top_k, "queries": []}

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        print(f"\n  [{qid}] {query_text}")

        # Embedding de la query
        query_vec = get_embedding(query_text)
        time.sleep(0.5)

        query_report = {"id": qid, "query": query_text, "type": q["type"],
                        "strategies": {}}

        for strategy_name, retrieval_fn in [
            ("vectorial", lambda: retrieve_vector(args.db, query_vec, args.top_k)),
            ("grafo", lambda: retrieve_graph(args.db, query_text, args.top_k)),
            ("hibrido", lambda: retrieve_hybrid(args.db, query_text, query_vec, args.top_k)),
            ("gnn", lambda: retrieve_gnn(args.db, query_vec, args.top_k)),
        ]:
            t0 = time.time()
            results = retrieval_fn()
            
            # Si la estrategia GNN devuelve vacío porque no existe el pt, la saltamos
            if strategy_name == "gnn" and not results:
                continue
                
            latency = round((time.time() - t0) * 1000, 1)

            coherence = metric_coherence(query_vec, results)
            diversity = metric_diversity_ils(results)
            coverage = metric_hotel_coverage(results)

            print(f"    {strategy_name:<12} "
                  f"coherencia={coherence:.3f}  "
                  f"diversidad={diversity:.3f}  "
                  f"hoteles={coverage}  "
                  f"latencia={latency}ms")

            # Guardar sin vectores (muy grandes)
            clean_results = []
            for r in results:
                cr = {k: v for k, v in r.items() if k != "vector"}
                clean_results.append(cr)

            query_report["strategies"][strategy_name] = {
                "coherence": coherence,
                "diversity": diversity,
                "hotel_coverage": coverage,
                "latency_ms": latency,
                "results": clean_results,
            }

        report["queries"].append(query_report)

    # Resumen agregado
    print(f"\n{'─' * 70}")
    print("  RESUMEN AGREGADO")
    print(f"{'─' * 70}")
    for strategy in ["vectorial", "grafo", "hibrido", "gnn"]:
        coherences = [q["strategies"][strategy]["coherence"]
                      for q in report["queries"] if strategy in q["strategies"]]
        diversities = [q["strategies"][strategy]["diversity"]
                       for q in report["queries"] if strategy in q["strategies"]]
        coverages = [q["strategies"][strategy]["hotel_coverage"]
                     for q in report["queries"] if strategy in q["strategies"]]
        latencies = [q["strategies"][strategy]["latency_ms"]
                     for q in report["queries"] if strategy in q["strategies"]]

        n = len(coherences)
        if n > 0:
            print(f"  {strategy:<12} "
                  f"coherencia={sum(coherences)/n:.3f}  "
                  f"diversidad={sum(diversities)/n:.3f}  "
                  f"hoteles={sum(coverages)/n:.1f}  "
                  f"latencia={sum(latencies)/n:.0f}ms")

    # Guardar
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"retrieval_report_{args.db}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte guardado en: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
