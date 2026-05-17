"""
GNN para Optimización de Búsquedas de Hoteles
==============================================
Fase 2 y 3: Diseño y Entrenamiento del Modelo.

Arquitectura: HeteroConv con SAGEConv.
  - Se añaden aristas inversas (Chunk->Hotel, Review->Hotel, etc.) para que el
    mensaje fluya HACIA los hoteles en cada capa de propagación.
  - Hotel inicia con el promedio de los embeddings de sus Chunks (warm start).

Loss: BPR (Bayesian Personalized Ranking) contrastiva auto-supervisada.

Uso:
    python gnn_model.py
    python gnn_model.py --db hotels_rvs_srvcs_graph_llm --epochs 150
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "gnn")


# ── Utilidades ───────────────────────────────────────────────────────────────

def add_reverse_edges(data: HeteroData) -> HeteroData:
    """
    Añade aristas inversas para cada tipo de arista existente.
    Ej. (Hotel, HAS_CHUNK, Chunk) -> (Chunk, rev_HAS_CHUNK, Hotel).
    Esto es imprescindible para que los nodos Hotel reciban mensajes.
    """
    new_edges = {}
    for (src, rel, dst), store in data.edge_index_dict.items():
        rev_name = f"rev_{rel}"
        rev_key = (dst, rev_name, src)
        if rev_key not in data.edge_index_dict:
            # Transponer el edge_index para invertir la dirección
            new_edges[rev_key] = store.flip(0)

    for key, ei in new_edges.items():
        data[key].edge_index = ei

    return data


def warm_start_hotels(data: HeteroData) -> HeteroData:
    """
    Inicializa los embeddings de Hotel como el promedio de los Chunks conectados.
    Mucho mejor que ceros como punto de partida.
    """
    key = ("Hotel", "HAS_CHUNK", "Chunk")
    if key not in data.edge_index_dict:
        return data  # Sin chunks, dejamos ceros

    ei = data[key].edge_index       # [2, num_edges]  src=hotel, dst=chunk
    hotel_idx = ei[0]
    chunk_idx  = ei[1]

    num_hotels = data['Hotel'].num_nodes
    embed_dim  = data['Chunk'].x.shape[1]

    # Sumar embeddings de chunks por hotel
    hotel_x = torch.zeros(num_hotels, embed_dim)
    hotel_x.index_add_(0, hotel_idx, data['Chunk'].x[chunk_idx])

    # Contar chunks por hotel para promediar
    counts = torch.zeros(num_hotels)
    counts.index_add_(0, hotel_idx, torch.ones(hotel_idx.size(0)))
    counts = counts.clamp(min=1).unsqueeze(1)

    data['Hotel'].x = hotel_x / counts
    return data


# ── Modelo ───────────────────────────────────────────────────────────────────

class HotelGNN(torch.nn.Module):
    """
    Dos capas de HeteroConv con SAGEConv, construidas dinámicamente según
    los tipos de arista (incluyendo los inversos).
    """
    def __init__(self, edge_types: list, embed_dim: int, hidden: int):
        super().__init__()
        self.conv1 = HeteroConv(
            {et: SAGEConv(embed_dim, hidden) for et in edge_types},
            aggr='mean'
        )
        self.conv2 = HeteroConv(
            {et: SAGEConv(hidden, embed_dim) for et in edge_types},
            aggr='mean'
        )

    def forward(self, x_dict, edge_index_dict):
        # Capa 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items() if v is not None}
        # Capa 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def bpr_loss(h_src, h_dst_pos, h_dst_neg):
    """BPR: acerca al hotel con sus vecinos reales, aleja con los aleatorios."""
    pos = (h_src * h_dst_pos).sum(dim=1)
    neg = (h_src * h_dst_neg).sum(dim=1)
    return -F.logsigmoid(pos - neg).mean()


def train_epoch(model, data, optimizer, target_edges, device):
    model.train()
    optimizer.zero_grad()

    out = model(data.x_dict, data.edge_index_dict)

    total_loss = torch.tensor(0.0, device=device)
    n = 0

    for et in target_edges:
        if et not in data.edge_index_dict:
            continue
        src_type, _, dst_type = et
        ei = data[et].edge_index
        src_idx, dst_idx_pos = ei[0], ei[1]

        if out.get(src_type) is None or out.get(dst_type) is None:
            continue

        h_src     = out[src_type][src_idx]
        h_dst_pos = out[dst_type][dst_idx_pos]

        # Negative sampling
        num_dst = data[dst_type].num_nodes
        dst_idx_neg = torch.randint(0, num_dst, (src_idx.size(0),), device=device)
        h_dst_neg = out[dst_type][dst_idx_neg]

        total_loss = total_loss + bpr_loss(h_src, h_dst_pos, h_dst_neg)
        n += 1

    if n > 0:
        (total_loss / n).backward()
        optimizer.step()

    return (total_loss / max(n, 1)).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(db_name, epochs=100, lr=0.005, hidden=256):
    print(f"\n{'=' * 60}")
    print(f"  GNN Training — {db_name}")
    print(f"{'=' * 60}")

    data_path = os.path.join(DATA_DIR, f"{db_name}_heterodata.pt")
    if not os.path.exists(data_path):
        print(f"ERROR: No se encontró {data_path}")
        print(f"  Ejecuta primero: python gnn_dataset_builder.py --db {db_name}")
        return

    data = torch.load(data_path, weights_only=False)

    embed_dim = data['Chunk'].x.shape[1]
    print(f"\nNodos  : Hotel={data['Hotel'].num_nodes}  "
          f"Chunk={data['Chunk'].num_nodes}  Review={data['Review'].num_nodes}")
    print(f"Embedding dim : {embed_dim}")
    print(f"Aristas originales: {len(data.metadata()[1])}")

    # Warm-start de hotel embeddings y añadir aristas inversas
    data = warm_start_hotels(data)
    data = add_reverse_edges(data)

    print(f"Aristas (con inversas): {len(data.metadata()[1])}")

    edge_types = list(data.edge_index_dict.keys())
    for et in edge_types:
        n = data[et].edge_index.shape[1]
        print(f"  {et[0]} --[{et[1]}]--> {et[2]}  ({n:,})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    data = data.to(device)

    model = HotelGNN(edge_types, embed_dim, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Aristas objetivo para BPR (Hotel como fuente)
    target_edges = [
        ("Hotel", "HAS_CHUNK",  "Chunk"),
        ("Hotel", "HAS_REVIEW", "Review"),
    ]

    print(f"\nEntrenando {epochs} épocas con BPR Loss...\n")
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, target_edges, device)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}/{epochs}  Loss={loss:.5f}  Best={best_loss:.5f}")

    print("\nEntrenamiento completado.")

    # Guardar modelo y embeddings
    model_path = os.path.join(DATA_DIR, f"{db_name}_gnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Modelo       : {model_path}")

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        hotel_embs = out['Hotel'].cpu()

    emb_path = os.path.join(DATA_DIR, f"{db_name}_gnn_embeddings.pt")
    torch.save(hotel_embs, emb_path)
    print(f"  Embeddings   : {emb_path}  shape={tuple(hotel_embs.shape)}")
    print("\nListos para usar en eval_retrieval.py y eval_rag_comparison.py (estrategia 'gnn').")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db",     default="hotels_rvs_srvcs_graph")
    p.add_argument("--epochs", type=int,   default=100)
    p.add_argument("--lr",     type=float, default=0.005)
    p.add_argument("--hidden", type=int,   default=256)
    a = p.parse_args()
    main(a.db, a.epochs, a.lr, a.hidden)
