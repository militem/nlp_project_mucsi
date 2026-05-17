"""
Comparación RAG Vectorial vs GraphRAG Híbrido
===============================================
Genera respuestas con ambos sistemas y las evalúa con LLM-as-Judge:
- Faithfulness (1-5): ¿la respuesta se basa en el contexto?
- Answer Relevancy (1-5): ¿responde la pregunta?
- Context Relevancy (1-5): ¿el contexto recuperado es útil?
- Completeness (1-5): ¿cubre todos los aspectos?

Uso:
    python eval_rag_comparison.py
    python eval_rag_comparison.py --db hotels_rvs_srvcs_graph
"""

import argparse
import json
import math
import os
import re
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

LM_STUDIO_URL = "http://192.168.1.151:1234"    # LLM remoto (chat/NER/judge)
EMBEDDING_URL = "http://localhost:1234"          # Embeddings local

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
QUERIES_FILE = os.path.join(SCRIPT_DIR, "queries_test.json")

TOP_K = 5

JUDGE_PROMPT = """Eres un evaluador experto de sistemas de búsqueda de hoteles.

PREGUNTA del usuario:
{query}

CONTEXTO recuperado por el sistema:
{context}

RESPUESTA generada:
{answer}

Evalúa la respuesta con puntuaciones del 1 al 5 en cada dimensión.
Devuelve SOLO un JSON con este formato exacto:
{{"faithfulness": X, "answer_relevancy": X, "context_relevancy": X, "completeness": X, "answer_found": true/false}}

Criterios:
- faithfulness: ¿La respuesta se basa SOLO en el contexto dado? (1=inventa datos, 5=todo verificable)
- answer_relevancy: ¿La respuesta contesta lo que pregunta el usuario con una recomendación concreta? (1=no responde o dice que no encontró nada, 5=recomienda hoteles específicos con justificación)
- context_relevancy: ¿El contexto recuperado contiene información útil y relevante para responder la pregunta? (1=ningún resultado es relevante, 5=todos los resultados son relevantes)
- completeness: ¿La respuesta cubre todos los aspectos de la pregunta? (1=no da ninguna recomendación, 5=respuesta exhaustiva con detalles)
- answer_found: ¿El sistema logró recomendar al menos un hotel concreto? (true/false). Si la respuesta dice "no encontré", "no hay hoteles", o similar → false

IMPORTANTE: Si la respuesta dice que no encontró resultados o no puede responder, answer_relevancy y completeness deben ser 1.

Devuelve SOLO el JSON, sin explicación."""

RAG_SYSTEM_PROMPT = """Eres un asistente experto en recomendación de hoteles.
Responde SOLO basándote en el contexto proporcionado.
Si no tienes información suficiente, dilo explícitamente.
Responde en español, de forma concisa y útil."""


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
        json={"input": text}, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def flatten_vector(vec) -> list:
    """Aplana vectores de ArcadeDB que vienen como [[float, ...]]."""
    if not vec:
        return []
    if isinstance(vec[0], list):
        return vec[0]
    return vec


def llm_generate(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    """Genera respuesta con el LLM."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_judge_scores(content: str) -> dict:
    """Parsea la respuesta del juez LLM."""
    content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "faithfulness": min(5, max(1, int(data.get("faithfulness", 3)))),
                "answer_relevancy": min(5, max(1, int(data.get("answer_relevancy", 3)))),
                "context_relevancy": min(5, max(1, int(data.get("context_relevancy", 3)))),
                "completeness": min(5, max(1, int(data.get("completeness", 3)))),
                "answer_found": bool(data.get("answer_found", False)),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"faithfulness": 0, "answer_relevancy": 0,
            "context_relevancy": 0, "completeness": 0, "answer_found": False}


# RAG Pipelines ──────────────────────────────────────────────────────────────

def rag_vectorial(db: str, query: str, query_vec: list, top_k: int) -> dict:
    """RAG vectorial puro: cosine similarity + keyword boost."""
    # Extraer keywords de la query para boost
    stop_words = {"hotel", "con", "en", "de", "y", "el", "la", "los", "las",
                  "un", "una", "para", "por", "qué", "que", "mejor", "vs",
                  "comparar", "tiene", "buenas", "opiniones", "sobre",
                  "cerca", "más", "del"}
    query_kws = {w for w in query.lower().split()
                 if w not in stop_words and len(w) > 2}

    all_items = []
    for vtype, id_field, edge in [("Chunk", "chunk_id", "HAS_CHUNK"),
                                    ("Review", "review_id", "HAS_REVIEW")]:
        try:
            r = arcadedb_query(db,
                f"SELECT {id_field}, text, vector, "
                f"in('{edge}')[0].name as hotel_name "
                f"FROM {vtype}")
            for item in r.get("result", []):
                vec = flatten_vector(item.get("vector", []))
                if not vec:
                    continue
                sim = cosine_sim(query_vec, vec)

                # Keyword boost: subir score si el texto contiene keywords
                text_lower = item.get("text", "").lower()
                kw_hits = sum(1 for kw in query_kws if kw in text_lower)
                boost = min(kw_hits * 0.05, 0.15)  # max +0.15
                sim_boosted = min(sim + boost, 1.0)

                all_items.append({
                    "text": item.get("text", ""),
                    "hotel": item.get("hotel_name", ""),
                    "sim": sim_boosted,
                })
        except Exception:
            pass

    all_items.sort(key=lambda x: x["sim"], reverse=True)
    top_items = all_items[:top_k]

    # Construir contexto
    context = "\n\n".join(
        f"[Hotel: {item['hotel']}] {item['text']}"
        for item in top_items
    )

    # Generar respuesta
    prompt = f"Contexto:\n{context}\n\nPregunta: {query}"
    answer = llm_generate(prompt, system=RAG_SYSTEM_PROMPT)

    return {"context": context, "answer": answer, "n_results": len(top_items)}


def rag_graph_hybrid(db: str, query: str, query_vec: list, top_k: int) -> dict:
    """GraphRAG híbrido: vector + metadatos del grafo."""
    # 1. Buscar por vector
    all_items = []
    for vtype, id_field, edge in [("Chunk", "chunk_id", "HAS_CHUNK"),
                                    ("Review", "review_id", "HAS_REVIEW")]:
        try:
            r = arcadedb_query(db,
                f"SELECT {id_field}, text, vector, "
                f"in('{edge}')[0].hotel_id as hotel_id, "
                f"in('{edge}')[0].name as hotel_name "
                f"FROM {vtype}")
            for item in r.get("result", []):
                vec = flatten_vector(item.get("vector", []))
                if vec:
                    sim = cosine_sim(query_vec, vec)
                    all_items.append({
                        "text": item.get("text", ""),
                        "hotel_id": item.get("hotel_id", ""),
                        "hotel": item.get("hotel_name", ""),
                        "sim": sim,
                    })
        except Exception:
            pass

    all_items.sort(key=lambda x: x["sim"], reverse=True)
    top_hotel_ids = []
    seen = set()
    for item in all_items:
        hid = item.get("hotel_id", "")
        if hid and hid not in seen:
            seen.add(hid)
            top_hotel_ids.append(hid)
        if len(top_hotel_ids) >= top_k:
            break

    # 2. Enriquecer con metadatos del grafo
    enriched_contexts = []
    for hid in top_hotel_ids:
        try:
            r = arcadedb_query(db,
                "SELECT name, city, rating, price, "
                "out('HAS_AMENITY').name as amenities, "
                "out('NEAR_LOCATION').name as locations, "
                "out('IN_PRICE_RANGE').label as price_range "
                f"FROM Hotel WHERE hotel_id = :hid",
                {"hid": hid})
            if r.get("result"):
                h = r["result"][0]
                # Coger el mejor chunk de este hotel
                hotel_text = next(
                    (it["text"] for it in all_items if it.get("hotel_id") == hid),
                    ""
                )
                ctx = (
                    f"Hotel: {h.get('name', '')} ({h.get('city', '')})\n"
                    f"  Rating: {h.get('rating', 'N/A')}/10 | "
                    f"Precio: {h.get('price', 'N/A')}€\n"
                    f"  Amenities: {', '.join(h.get('amenities', [])[:10])}\n"
                    f"  Ubicaciones cercanas: {', '.join(h.get('locations', [])[:5])}\n"
                    f"  Rango precio: {h.get('price_range', 'N/A')}\n"
                    f"  Info: {hotel_text[:300]}"
                )
                enriched_contexts.append(ctx)
        except Exception:
            pass

    context = "\n\n---\n\n".join(enriched_contexts)

    # 3. Generar respuesta
    prompt = f"Contexto:\n{context}\n\nPregunta: {query}"
    answer = llm_generate(prompt, system=RAG_SYSTEM_PROMPT)

    return {"context": context, "answer": answer, "n_results": len(enriched_contexts)}


def rag_gnn(db: str, query: str, query_vec: list, top_k: int) -> dict:
    """GraphRAG usando los embeddings enriquecidos de la GNN para encontrar hoteles."""
    if torch is None:
        return {"context": "", "answer": "Error: PyTorch no disponible", "n_results": 0}
        
    data_dir = os.path.join(SCRIPT_DIR, "..", "data", "gnn")
    emb_path = os.path.join(data_dir, f"{db}_gnn_embeddings.pt")
    map_path = os.path.join(data_dir, f"{db}_mappings.json")
    
    if not os.path.exists(emb_path) or not os.path.exists(map_path):
        return {"context": "", "answer": "Error: Embeddings GNN no encontrados", "n_results": 0}
        
    gnn_embs = torch.load(emb_path, map_location='cpu', weights_only=False) # [num_hotels, emb_dim]
    with open(map_path, "r") as f:
        mappings = json.load(f) # {idx_str: hotel_id}
        
    q_tensor = torch.tensor([query_vec], dtype=torch.float32)
    sims = F.cosine_similarity(q_tensor, gnn_embs)
    
    top_scores, top_indices = torch.topk(sims, min(top_k * 2, len(sims)))
    
    top_hotel_ids = []
    seen = set()
    for idx in top_indices.tolist():
        hid = mappings.get(str(idx))
        if hid and hid not in seen:
            seen.add(hid)
            top_hotel_ids.append(hid)
        if len(top_hotel_ids) >= top_k:
            break

    # Enriquecer con metadatos del grafo (mismo que GraphRAG pero usando estos hoteles)
    enriched_contexts = []
    for hid in top_hotel_ids:
        try:
            r = arcadedb_query(db,
                "SELECT name, city, rating, price, "
                "out('HAS_AMENITY').name as amenities, "
                "out('NEAR_LOCATION').name as locations, "
                "out('IN_PRICE_RANGE').label as price_range "
                f"FROM Hotel WHERE hotel_id = :hid",
                {"hid": hid})
            if r.get("result"):
                h = r["result"][0]
                
                # Fetch a chunk from the hotel to give text context
                hotel_text = ""
                r_chunk = arcadedb_query(db, 
                    f"SELECT text FROM Chunk WHERE in('HAS_CHUNK')[0].hotel_id = :hid LIMIT 1",
                    {"hid": hid})
                if r_chunk.get("result"):
                    hotel_text = r_chunk["result"][0].get("text", "")
                
                ctx = (
                    f"Hotel: {h.get('name', '')} ({h.get('city', '')})\n"
                    f"  Rating: {h.get('rating', 'N/A')}/10 | "
                    f"Precio: {h.get('price', 'N/A')}€\n"
                    f"  Amenities: {', '.join(h.get('amenities', [])[:10])}\n"
                    f"  Ubicaciones cercanas: {', '.join(h.get('locations', [])[:5])}\n"
                    f"  Rango precio: {h.get('price_range', 'N/A')}\n"
                    f"  Info: {hotel_text[:300]}"
                )
                enriched_contexts.append(ctx)
        except Exception:
            pass

    context = "\n\n---\n\n".join(enriched_contexts)
    prompt = f"Contexto:\n{context}\n\nPregunta: {query}"
    answer = llm_generate(prompt, system=RAG_SYSTEM_PROMPT)

    return {"context": context, "answer": answer, "n_results": len(enriched_contexts)}

# Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparación RAG vs GraphRAG")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  COMPARACIÓN RAG: Vectorial vs GraphRAG vs GNN – {args.db}")
    print("=" * 70)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    report = {"database": args.db, "top_k": args.top_k, "queries": []}

    # Acumuladores para resumen
    totals = {
        "vectorial": {"faithfulness": [], "answer_relevancy": [],
                      "context_relevancy": [], "completeness": [],
                      "answer_found": []},
        "graphrag": {"faithfulness": [], "answer_relevancy": [],
                     "context_relevancy": [], "completeness": [],
                     "answer_found": []},
        "gnn": {"faithfulness": [], "answer_relevancy": [],
                "context_relevancy": [], "completeness": [],
                "answer_found": []},
    }

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        print(f"\n{'─' * 60}")
        print(f"  [{qid}] {query_text}")

        query_vec = get_embedding(query_text)
        time.sleep(0.5)

        query_report = {"id": qid, "query": query_text, "systems": {}}

        for sys_name, rag_fn in [
            ("vectorial", lambda: rag_vectorial(args.db, query_text, query_vec, args.top_k)),
            ("graphrag", lambda: rag_graph_hybrid(args.db, query_text, query_vec, args.top_k)),
            ("gnn", lambda: rag_gnn(args.db, query_text, query_vec, args.top_k)),
        ]:
            # Si el pt no existe y devuelve Error, lo saltamos
            t0 = time.time()
            result = rag_fn()
            
            if sys_name == "gnn" and result["n_results"] == 0 and "Error" in result["answer"]:
                continue
                
            print(f"\n    {sys_name.upper()}:")
            latency = round((time.time() - t0) * 1000, 1)

            print(f"      Respuesta: {result['answer'][:120]}...")
            print(f"      Resultados: {result['n_results']} | Latencia: {latency}ms")

            # LLM-as-Judge
            judge_prompt = JUDGE_PROMPT.format(
                query=query_text,
                context=result["context"][:2000],
                answer=result["answer"],
            )
            time.sleep(1)
            judge_response = llm_generate(judge_prompt, max_tokens=200)
            scores = parse_judge_scores(judge_response)

            print(f"      Scores: faith={scores['faithfulness']} "
                  f"relev={scores['answer_relevancy']} "
                  f"ctx={scores['context_relevancy']} "
                  f"compl={scores['completeness']} "
                  f"found={scores['answer_found']}")

            query_report["systems"][sys_name] = {
                "answer": result["answer"],
                "context_length": len(result["context"]),
                "n_results": result["n_results"],
                "latency_ms": latency,
                "scores": scores,
            }

            for metric in ["faithfulness", "answer_relevancy",
                           "context_relevancy", "completeness"]:
                val = scores[metric]
                if val > 0:
                    totals[sys_name][metric].append(val)
            totals[sys_name]["answer_found"].append(
                1 if scores["answer_found"] else 0)

            time.sleep(1)

        report["queries"].append(query_report)

    print(f"\n{'=' * 75}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'=' * 75}")
    print(f"  {'Métrica':<25} {'Vectorial':>10} {'GraphRAG':>10} {'GNN':>10} {'Ganador':>10}")
    print(f"  {'─' * 65}")

    summary = {}
    for metric in ["faithfulness", "answer_relevancy", "context_relevancy", "completeness"]:
        vec_vals = totals["vectorial"][metric]
        graph_vals = totals["graphrag"][metric]
        gnn_vals = totals["gnn"][metric]
        
        vec_avg = sum(vec_vals) / len(vec_vals) if vec_vals else 0
        graph_avg = sum(graph_vals) / len(graph_vals) if graph_vals else 0
        gnn_avg = sum(gnn_vals) / len(gnn_vals) if gnn_vals else 0
        
        best_avg = max(vec_avg, graph_avg, gnn_avg)
        if best_avg == gnn_avg and gnn_avg > 0: winner = "GNN"
        elif best_avg == graph_avg and graph_avg > 0: winner = "GraphRAG"
        elif best_avg == vec_avg and vec_avg > 0: winner = "Vectorial"
        else: winner = "Empate"

        print(f"  {metric:<25} {vec_avg:>10.2f} {graph_avg:>10.2f} {gnn_avg:>10.2f} {winner:>10}")
        summary[metric] = {"vectorial": round(vec_avg, 2),
                           "graphrag": round(graph_avg, 2),
                           "gnn": round(gnn_avg, 2),
                           "winner": winner}

    # Hit rate (answer_found)
    for sys_name in ["vectorial", "graphrag", "gnn"]:
        found = totals[sys_name]["answer_found"]
        hit_rate = sum(found) / len(found) * 100 if found else 0
        summary[f"{sys_name}_hit_rate"] = round(hit_rate, 1)
    print(f"\n  {'Hit Rate (answer_found)':<25} "
          f"{summary['vectorial_hit_rate']:>9.1f}% "
          f"{summary['graphrag_hit_rate']:>9.1f}% "
          f"{summary['gnn_hit_rate']:>9.1f}%")

    report["summary"] = summary

    # Guardar
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"rag_comparison_{args.db}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Reporte guardado en: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
