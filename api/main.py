"""
API FastAPI - Busqueda hibrida de hoteles
==========================================
Endpoint para busqueda de hoteles en lenguaje natural.
Combina NER + grafo (ArcadeDB) + vectores (cosine similarity).

Uso:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /api/search  -  Busqueda hibrida, devuelve top 5 hoteles
    GET  /api/health   -  Estado de la API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from hybrid_search import HybridSearchEngine

# Motor de busqueda (singleton)
engine = HybridSearchEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el motor de busqueda al iniciar la API."""
    engine.load()
    yield


app = FastAPI(
    title="Hotel Hybrid Search API",
    description="Busqueda hibrida de hoteles combinando NER, grafos y vectores",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS para permitir conexion desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Modelos de request/response ------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    top_k: int = Field(default=6, ge=1, le=20, description="Numero de resultados")


class HotelResult(BaseModel):
    hotel_id: str
    name: str
    city: str
    address: str
    rating: float | None
    price: float | None
    services: str
    url: str
    source: str
    score: float
    graph_score: float
    vector_score: float
    matched_entities: list[str]


class SearchResponse(BaseModel):
    query: str
    entities_detected: list[dict]
    search_time_s: float
    total_candidates: int
    rag_answer: str | None = None
    results: list[HotelResult]


# -- Endpoints ------------------------------------------------------------

@app.post("/api/search", response_model=SearchResponse)
async def search_hotels(request: SearchRequest):
    """
    Busqueda hibrida de hoteles en lenguaje natural.

    Flujo:
    1. NER sobre la query para extraer entidades (ciudades, servicios, etc.)
    2. Busqueda por grafo: entidades -> traversal en ArcadeDB -> hoteles
    3. Busqueda vectorial: embedding de la query -> cosine similarity -> hoteles
    4. Combinacion de scores y devolucion del top K
    """
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Motor de busqueda no inicializado")

    try:
        result = engine.search(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la busqueda: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Estado de la API y sus dependencias."""
    return {
        "status": "ok" if engine.ready else "loading",
        "hotels_loaded": len(engine.hotels_data),
        "chunks_loaded": len(engine.chunk_vectors),
    }
