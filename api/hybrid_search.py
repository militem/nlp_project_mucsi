"""
Motor de busqueda hibrida: Grafo (ArcadeDB) + Vectores (cosine similarity)
===========================================================================
1. Procesa la query del usuario con NER (spaCy) para extraer entidades
2. Busqueda por grafo: entidades NER -> nodos Entity/City -> traversal -> Hotels
3. Busqueda vectorial: embedding de la query -> cosine similarity vs chunks
4. Combina ambos scores y devuelve top K hoteles

Requisitos:
    pip install requests spacy numpy
"""

import json
import os
import time
import numpy as np
import requests
import spacy
from collections import defaultdict
from typing import Optional

# Configuracion ──────────────────────────────────────────────────────────────

ARCADEDB_URL = "http://localhost:2480"
ARCADEDB_USER = "root"
ARCADEDB_PASSWORD = "hotel_nlp_2024"
DATABASE_NAME = "hotels_rvs_srvcs_graph"

LM_STUDIO_URL = "http://127.0.0.1:1234"

# Pesos para el scoring hibrido
ALPHA_GRAPH = 0.5
ALPHA_VECTOR = 0.5

TOP_K = 6

# Ruta al archivo de embeddings (para cargar vectores directamente)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(
    os.path.dirname(SCRIPT_DIR), "procesamiento", "hotels_embeddings_reviews_services.json"
)


class HybridSearchEngine:
    """Motor de busqueda hibrida que combina grafo y vectores."""

    def __init__(self):
        self.nlp_es = None
        self.nlp_en = None
        self.hotels_data = {}       # hotel_id -> dict con todos los datos
        self.chunk_vectors = {}     # chunk_id -> np.array
        self.chunk_to_hotel = {}    # chunk_id -> hotel_id
        self.ready = False

    def load(self):
        """Carga modelos NER y datos de hoteles/vectores."""
        print("Cargando motor de busqueda hibrida...")

        # Cargar modelos spaCy
        print("  Cargando modelos NER...")
        self.nlp_es = spacy.load("es_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm")
        print("  Modelos NER cargados")

        # Cargar datos de hoteles y vectores desde el archivo local
        # (mas rapido que traerlos por REST desde ArcadeDB)
        print(f"  Cargando datos desde {EMBEDDINGS_FILE}...")
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            hotels = json.load(f)

        for hotel in hotels:
            hid = hotel["hotel_id"]
            services_raw = hotel.get("services", "")
            if isinstance(services_raw, list):
                services_str = ", ".join(services_raw)
            else:
                services_str = str(services_raw)

            try:
                rating = float(hotel.get("rating"))
            except (ValueError, TypeError):
                rating = None
                
            try:
                price = float(hotel.get("price"))
            except (ValueError, TypeError):
                price = None

            self.hotels_data[hid] = {
                "hotel_id": hid,
                "name": hotel.get("name", ""),
                "city": hotel.get("city", ""),
                "address": hotel.get("address", ""),
                "rating": rating,
                "price": price,
                "services": services_str,
                "url": hotel.get("url", ""),
                "source": hotel.get("source", ""),
            }
            for chunk in hotel.get("chunks", []):
                cid = chunk["chunk_id"]
                vec = chunk.get("vector")
                if vec:
                    self.chunk_vectors[cid] = np.array(vec, dtype=np.float32)
                    self.chunk_to_hotel[cid] = hid

        print(f"  {len(self.hotels_data)} hoteles, {len(self.chunk_vectors)} chunks cargados")
        self.ready = True

    # -- Procesamiento de la query ----------------------------------------

    def process_query(self, query: str):
        """Procesa la query con NER y genera embedding."""
        # NER con ambos modelos para mayor cobertura
        raw_entities = []
        seen = set()
        for nlp in [self.nlp_es, self.nlp_en]:
            doc = nlp(query)
            for ent in doc.ents:
                name = ent.text.strip()
                key = name.lower()
                if key not in seen and len(name) > 1:
                    seen.add(key)
                    raw_entities.append({"name": name, "label": ent.label_})

        # Filtrar entidades ruidosas:
        # - "hotel en madrid" contiene "madrid" que ya es entidad propia
        # - Entidades que empiezan con palabras genericas (hotel, quiero, busco)
        noise_prefixes = ("hotel ", "quiero ", "busco ", "necesito ")
        entity_names = {e["name"].lower() for e in raw_entities}
        entities = []
        for ent in raw_entities:
            name_lower = ent["name"].lower()
            # Descartar si empieza con prefijo generico y contiene una entidad ya detectada
            if any(name_lower.startswith(p) for p in noise_prefixes):
                # Comprobar si la parte relevante ya existe como entidad propia
                has_overlap = any(
                    other != name_lower and other in name_lower
                    for other in entity_names
                )
                if has_overlap:
                    continue
            entities.append(ent)

        # Embedding de la query via LM Studio
        query_vector = self._get_embedding(query)

        return entities, query_vector

    def graph_search(self, entities: list, query: str = "") -> dict:
        """Busca hoteles usando el grafo enriquecido.

        Estrategia de scoring:
        - Entidad que matchea Ciudad -> +2 a todos los hoteles de esa ciudad
        - Entidad que matchea DIRECCION -> +5 (match mas fuerte)
        - Entidad que matchea NOMBRE del hotel -> +4
        - Entidad que matchea Amenity (via HAS_AMENITY) -> +3
        - Entidad que matchea Location (via NEAR_LOCATION) -> +3
        """
        hotel_scores = defaultdict(float)

        if not entities and not query:
            return hotel_scores

        for ent in entities:
            ent_name = ent["name"].lower()

            # 1. Ver si es una ciudad
            is_city = False
            try:
                city_result = self._db_query(
                    "SELECT hotel_id FROM Hotel WHERE city.toLowerCase() LIKE :city",
                    {"city": f"%{ent_name}%"},
                )
                city_hotels = city_result.get("result", [])
                if city_hotels:
                    is_city = True
                    for rec in city_hotels:
                        hid = rec.get("hotel_id")
                        if hid:
                            hotel_scores[hid] += 2.0
            except Exception:
                pass

            if not is_city:
                # 2. Match DIRECTO en direccion
                try:
                    result = self._db_query(
                        "SELECT hotel_id FROM Hotel WHERE address.toLowerCase() LIKE :addr",
                        {"addr": f"%{ent_name}%"},
                    )
                    for rec in result.get("result", []):
                        hid = rec.get("hotel_id")
                        if hid:
                            hotel_scores[hid] += 5.0
                except Exception:
                    pass

                # 3. Match DIRECTO en nombre del hotel
                try:
                    result = self._db_query(
                        "SELECT hotel_id FROM Hotel WHERE name.toLowerCase() LIKE :name",
                        {"name": f"%{ent_name}%"},
                    )
                    for rec in result.get("result", []):
                        hid = rec.get("hotel_id")
                        if hid:
                            hotel_scores[hid] += 4.0
                except Exception:
                    pass

            # 4. Buscar en Amenity -> HAS_AMENITY -> Hotel
            try:
                result = self._db_query(
                    "SELECT expand(in('HAS_AMENITY')) FROM Amenity "
                    "WHERE name.toLowerCase() LIKE :name",
                    {"name": f"%{ent_name}%"},
                )
                for rec in result.get("result", []):
                    hid = rec.get("hotel_id")
                    if hid:
                        hotel_scores[hid] += 3.0
            except Exception:
                pass

            # 5. Buscar en Location -> NEAR_LOCATION -> Hotel
            try:
                result = self._db_query(
                    "SELECT expand(in('NEAR_LOCATION')) FROM Location "
                    "WHERE name.toLowerCase() LIKE :name",
                    {"name": f"%{ent_name}%"},
                )
                for rec in result.get("result", []):
                    hid = rec.get("hotel_id")
                    if hid:
                        hotel_scores[hid] += 3.0
            except Exception:
                pass

        # Normalizar a [0, 1]
        if hotel_scores:
            max_score = max(hotel_scores.values())
            if max_score > 0:
                for k in hotel_scores:
                    hotel_scores[k] /= max_score

        return hotel_scores

    # -- Busqueda vectorial -----------------------------------------------

    def vector_search(self, query_vector, top_k: int = 50) -> dict:
        """Busca por similitud coseno del vector query vs chunks."""
        if query_vector is None or not self.chunk_vectors:
            return {}

        qv = np.array(query_vector, dtype=np.float32)
        qv_norm = np.linalg.norm(qv)
        if qv_norm < 1e-10:
            return {}
        qv = qv / qv_norm

        # Calcular similitud coseno con todos los chunks
        chunk_ids = list(self.chunk_vectors.keys())
        vectors = np.stack([self.chunk_vectors[cid] for cid in chunk_ids])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vectors_norm = vectors / norms

        similarities = vectors_norm @ qv  # (N,)

        # Agregar por hotel (max similarity por hotel)
        hotel_scores = defaultdict(float)
        for i, cid in enumerate(chunk_ids):
            sim = float(similarities[i])
            hid = self.chunk_to_hotel.get(cid)
            if hid:
                hotel_scores[hid] = max(hotel_scores[hid], sim)

        # Normalizar a [0, 1]
        if hotel_scores:
            values = list(hotel_scores.values())
            min_s = min(values)
            max_s = max(values)
            range_s = max_s - min_s if max_s != min_s else 1.0
            for k in hotel_scores:
                hotel_scores[k] = (hotel_scores[k] - min_s) / range_s

        return hotel_scores

    # -- Busqueda hibrida -------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K) -> dict:
        """Busqueda hibrida: grafo + vectores. Devuelve top K hoteles."""
        t0 = time.time()

        entities, query_vector = self.process_query(query)

        # Generar embeddings de query
        query_vector = self._get_embedding(query)
        if not query_vector:
            # Fallback a un vector vacio si LM Studio falla
            query_vector = [0.0] * 384
        
        # 2. Vector Search (In-memory for stability and speed as evaluated in eval_rag_comparison.py)
        v_scores = defaultdict(float)
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        if q_norm > 0:
            chunk_sims = []
            for cid, c_vec in self.chunk_vectors.items():
                c_norm = np.linalg.norm(c_vec)
                if c_norm > 0:
                    dot = np.dot(q_vec, c_vec)
                    sim = dot / (q_norm * c_norm)
                    chunk_sims.append((cid, float(sim)))
            
            # Poda del arbol: Solo tomamos los top 50 chunks más relevantes
            chunk_sims.sort(key=lambda x: x[1], reverse=True)
            for cid, sim in chunk_sims[:50]:
                hid = self.chunk_to_hotel[cid]
                v_scores[hid] = max(v_scores[hid], sim)

        # 3. Graph Search (NER)
        # Buscar en el grafo con las entidades detectadas
        g_scores = self.graph_search(entities, query)

        # 4. Combinar scores
        all_hotel_ids = set(v_scores.keys()) | set(g_scores.keys())
        results = []

        for hid in all_hotel_ids:
            v_score = v_scores.get(hid, 0.0)
            g_score = g_scores.get(hid, 0.0)

            # Normalizar Graph Score (asumiendo maximo teorico ~10)
            norm_g = min(1.0, g_score / 10.0)

            # Score Hibrido
            hybrid = (ALPHA_VECTOR * v_score) + (ALPHA_GRAPH * norm_g)
            
            # Boost: Si ambas fuentes encuentran el hotel, bonificamos
            if v_score > 0 and g_score > 0:
                hybrid *= 1.2
                
            hotel_data = self.hotels_data.get(hid)
            if hotel_data:
                results.append({
                    **hotel_data,
                    "score": round(hybrid, 4),
                    "graph_score": round(g_score, 4),
                    "vector_score": round(v_score, 4),
                    "matched_entities": [
                        e["name"] for e in entities
                    ] if g_score > 0 else [],
                })

        # Ordenar por score descendente
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        
        # 5. GraphRAG: Generacion de Respuesta con el LLM
        rag_answer = ""
        if top_results:
            enriched_contexts = []
            for h in top_results:
                hid = h["hotel_id"]
                try:
                    r = self._db_query(
                        "SELECT name, city, rating, price, "
                        "out('HAS_AMENITY').name as amenities, "
                        "out('NEAR_LOCATION').name as locations "
                        "FROM Hotel WHERE hotel_id = :hid",
                        {"hid": hid}
                    )
                    if r.get("result"):
                        graph_data = r["result"][0]
                        # Obtener un texto contextual de un chunk
                        hotel_text = ""
                        r_chunk = self._db_query(
                            "SELECT text FROM Chunk WHERE in('HAS_CHUNK')[0].hotel_id = :hid LIMIT 1",
                            {"hid": hid}
                        )
                        if r_chunk.get("result"):
                            hotel_text = r_chunk["result"][0].get("text", "")
                            
                        ctx = (
                            f"Hotel: {graph_data.get('name', h.get('name', ''))} ({graph_data.get('city', h.get('city', ''))})\n"
                            f"  Rating: {graph_data.get('rating', h.get('rating', 'N/A'))}/10 | "
                            f"Precio: {graph_data.get('price', h.get('price', 'N/A'))}€\n"
                            f"  Amenities: {', '.join(graph_data.get('amenities', [])[:10])}\n"
                            f"  Ubicaciones cercanas: {', '.join(graph_data.get('locations', [])[:5])}\n"
                            f"  Info: {hotel_text[:300]}"
                        )
                        enriched_contexts.append(ctx)
                except Exception:
                    pass
            
            # Prompteamos al LLM con los contextos enriquecidos
            context_str = "\n\n---\n\n".join(enriched_contexts)
            prompt = f"Contexto:\n{context_str}\n\nPregunta: {query}"
            system_prompt = "Eres un experto en recomendación de hoteles. Responde de manera útil, concisa y amigable basándote SOLO en el contexto. Recomienda las opciones que mejor se ajusten."
            
            try:
                llm_resp = requests.post(
                    f"{LM_STUDIO_URL}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                    timeout=30
                )
                if llm_resp.ok:
                    content = llm_resp.json()["choices"][0]["message"]["content"]
                    # Limpiar thinking si lo hubiera
                    import re
                    content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                    rag_answer = content
            except Exception as e:
                print(f"Error generando respuesta RAG: {e}")

        elapsed = round(time.time() - t0, 3)
        return {
            "query": query,
            "entities_detected": entities,
            "search_time_s": elapsed,
            "total_candidates": len(all_hotel_ids),
            "rag_answer": rag_answer,
            "results": top_results,
        }

    # -- Helpers privados -------------------------------------------------

    def _db_query(self, command: str, params: Optional[dict] = None) -> dict:
        """Ejecuta un comando SQL en ArcadeDB."""
        payload = {"language": "sql", "command": command}
        if params:
            payload["params"] = params
        resp = requests.post(
            f"{ARCADEDB_URL}/api/v1/command/{DATABASE_NAME}",
            json=payload,
            auth=(ARCADEDB_USER, ARCADEDB_PASSWORD),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_embedding(self, text: str):
        """Obtiene el embedding de un texto via LM Studio."""
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL}/v1/embeddings",
                json={"input": text},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"  Aviso: no se pudo obtener embedding: {e}")
            return None
