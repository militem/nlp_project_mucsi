import os
import sys
import json
import torch
import requests
from torch_geometric.data import HeteroData

# Configuración ──────────────────────────────────────────────────────────────
ARCADEDB_URL = "http://localhost:2480"
ARCADEDB_USER = "root"
ARCADEDB_PASSWORD = "hotel_nlp_2024"
DEFAULT_DB = "hotels_rvs_srvcs_graph" # Puedes cambiar al de _llm o pasarlo por args
EMBEDDING_URL = "http://localhost:1234" # Para vectorizar amenities/cities si hace falta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "gnn")

def arcadedb_query(db: str, sql: str, params: dict = None) -> list:
    payload = {"language": "sql", "command": sql}
    if params:
        payload["params"] = params
    resp = requests.post(
        f"{ARCADEDB_URL}/api/v1/command/{db}",
        json=payload, auth=(ARCADEDB_USER, ARCADEDB_PASSWORD), timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("result", [])

def get_embedding(text: str) -> list:
    """Obtiene embedding local para nodos que solo tienen texto (Amenity, Location)."""
    resp = requests.post(
        f"{EMBEDDING_URL}/v1/embeddings",
        json={"input": text}, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

def flatten_vector(vec) -> list:
    if not vec: return []
    if isinstance(vec[0], list): return vec[0]
    return vec

def build_heterodata(db: str):
    print(f"Extrayendo grafo desde {db}...")
    data = HeteroData()

    # Mapeos globales de ID a índice continuo (0, 1, 2...) requerido por PyG
    # Mapeos de tipo: rid -> int_idx
    mappings = {
        "Hotel": {},
        "Chunk": {},
        "Review": {},
        "Amenity": {},
        "Location": {},
        "City": {}
    }

    # 1. Nodos Hotel
    print("  -> Cargando Hoteles...")
    hoteles = arcadedb_query(db, "SELECT @rid.asString() as rid, hotel_id, name FROM Hotel")
    for i, h in enumerate(hoteles):
        mappings["Hotel"][h["rid"]] = i
    num_hotels = len(hoteles)
    # Inicialmente los hoteles no tienen vector, usaremos un tensor de ceros 
    # o podríamos usar un promedio de sus chunks. De momento, un Embedding bag o ceros.
    # Necesitamos saber la dimensión del embedding. Asumiremos 1536 o 384 según el modelo.
    
    # 2. Nodos Chunk
    print("  -> Cargando Chunks...")
    chunks = arcadedb_query(db, "SELECT @rid.asString() as rid, vector FROM Chunk")
    chunk_vectors = []
    for i, c in enumerate(chunks):
        mappings["Chunk"][c["rid"]] = i
        vec = flatten_vector(c.get("vector", []))
        chunk_vectors.append(vec)
    
    if chunk_vectors:
        embed_dim = len(chunk_vectors[0])
        data['Chunk'].x = torch.tensor(chunk_vectors, dtype=torch.float)
    else:
        embed_dim = 1536 # Default fallback

    # Asignar ceros a Hotel por ahora (o embedding trainable)
    data['Hotel'].x = torch.zeros((num_hotels, embed_dim), dtype=torch.float)

    # 3. Nodos Review
    print("  -> Cargando Reviews...")
    reviews = arcadedb_query(db, "SELECT @rid.asString() as rid, vector FROM Review")
    review_vectors = []
    for i, r in enumerate(reviews):
        mappings["Review"][r["rid"]] = i
        vec = flatten_vector(r.get("vector", []))
        if not vec: vec = [0.0]*embed_dim # Padding si falla
        review_vectors.append(vec)
    if review_vectors:
        data['Review'].x = torch.tensor(review_vectors, dtype=torch.float)

    # 4. Amenities y Locations (No tienen vector en la BD, generamos uno on-the-fly)
    print("  -> Cargando Amenities...")
    amenities = arcadedb_query(db, "SELECT @rid.asString() as rid, name FROM Amenity")
    amenity_vectors = []
    for i, a in enumerate(amenities):
        mappings["Amenity"][a["rid"]] = i
        try:
            vec = get_embedding(a["name"])
        except:
            vec = [0.0]*embed_dim
        amenity_vectors.append(vec)
    if amenity_vectors:
        data['Amenity'].x = torch.tensor(amenity_vectors, dtype=torch.float)

    print("  -> Cargando Locations...")
    locations = arcadedb_query(db, "SELECT @rid.asString() as rid, name FROM Location")
    location_vectors = []
    for i, loc in enumerate(locations):
        mappings["Location"][loc["rid"]] = i
        try:
            vec = get_embedding(loc["name"])
        except:
            vec = [0.0]*embed_dim
        location_vectors.append(vec)
    if location_vectors:
        data['Location'].x = torch.tensor(location_vectors, dtype=torch.float)

    # 5. Cargar Aristas
    print("  -> Cargando Relaciones (Edges)...")
    
    # helper
    def load_edges(edge_class, src_type, dst_type):
        import re
        query = f"SELECT @rid.asString() as src, out('{edge_class}').@rid.asString() as dst_list FROM {src_type}"
        try:
            edges = arcadedb_query(db, query)
            src_indices = []
            dst_indices = []
            for e in edges:
                src_rid = e.get("src")
                dst_list_str = e.get("dst_list", "")
                if not dst_list_str or src_rid not in mappings[src_type]:
                    continue
                
                # Extraer rids formato #1:2
                dst_rids = re.findall(r'#\d+:\d+', dst_list_str)
                src_idx = mappings[src_type][src_rid]
                
                for dst_rid in dst_rids:
                    if dst_rid in mappings[dst_type]:
                        src_indices.append(src_idx)
                        dst_indices.append(mappings[dst_type][dst_rid])
            
            if src_indices:
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                data[src_type, edge_class, dst_type].edge_index = edge_index
                print(f"     {edge_class}: {len(src_indices)} aristas.")
            else:
                print(f"     {edge_class}: 0 aristas encontradas.")
        except Exception as e:
            print(f"     No se pudo cargar {edge_class}: {e}")

    load_edges("HAS_CHUNK", "Hotel", "Chunk")
    load_edges("HAS_REVIEW", "Hotel", "Review")
    load_edges("HAS_AMENITY", "Hotel", "Amenity")
    load_edges("NEAR_LOCATION", "Hotel", "Location")

    # Mostrar info del grafo
    print("\nResumen del Grafo (HeteroData):")
    print(data)

    # Guardar a disco
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"{db}_heterodata.pt")
    torch.save(data, out_file)
    print(f"\nGrafo guardado exitosamente en: {out_file}")
    
    # También guardamos los mappings para luego poder hacer el mapeo inverso (idx -> hotel_id)
    map_file = os.path.join(OUTPUT_DIR, f"{db}_mappings.json")
    # Limpiamos los mappings para que solo tengan el mapeo idx -> hotel_id
    hotel_map = {idx: rid for rid, idx in mappings["Hotel"].items()}
    # Buscamos los hotel_id reales
    hotel_id_map = {}
    for h in hoteles:
        idx = mappings["Hotel"][h["rid"]]
        hotel_id_map[idx] = h["hotel_id"]

    with open(map_file, "w") as f:
        json.dump(hotel_id_map, f, indent=2)
    
    print(f"Mappings guardados en: {map_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB, help="Nombre de la BD (ej. hotels_rvs_srvcs_graph o hotels_rvs_srvcs_graph_llm)")
    args = parser.parse_args()
    build_heterodata(args.db)
