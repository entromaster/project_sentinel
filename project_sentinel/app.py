"""
Core D — FastAPI Application + Add-ons E & F

Endpoints:
  GET  /health           → {"status": "ok"}
  GET  /books            → list books, optional ?category= filter
  GET  /books/{id}       → single book by ID, 404 if missing
  POST /classify         → predict category from {title, description}
  GET  /similar/{id}     → (Add-on E) 5 most similar books by vector similarity
  GET  /graph/book/{id}  → (Add-on F) book's neighbourhood in knowledge graph
"""

import os
import sqlite3
from contextlib import asynccontextmanager

import joblib
import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────
DB_FILE = "books.db"
MODEL_FILE = os.path.join("models", "classifier.pkl")
CHROMA_DIR = os.path.join("models", "chroma_db")

# ─── Globals (loaded at startup) ──────────────────────────────────────────────
classifier = None
chroma_collection = None
knowledge_graph = None


# ─── Pydantic Models ─────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str


class BookResponse(BaseModel):
    id: int
    upc: str
    title: str
    category: str
    description: str | None = None
    price: float
    star_rating: int
    availability: str
    url: str


class ClassifyRequest(BaseModel):
    title: str
    description: str = ""


class ClassifyResponse(BaseModel):
    predicted_category: str
    confidence: float


class SimilarBookResponse(BaseModel):
    id: int
    title: str
    category: str
    similarity_score: float


class GraphNeighbourhood(BaseModel):
    book: BookResponse
    same_category: list[dict]
    similar_books: list[dict]


# ─── Database helpers ─────────────────────────────────────────────────────────
def get_db():
    """Get a new SQLite connection with row factory."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_book(row: sqlite3.Row) -> dict:
    """Convert a DB row to a book dict."""
    return {
        "id": row["id"],
        "upc": row["upc"],
        "title": row["title"],
        "category": row["category_name"],
        "description": row["description"],
        "price": row["price"],
        "star_rating": row["star_rating"],
        "availability": row["availability"],
        "url": row["url"],
    }


BOOK_QUERY = """
    SELECT b.id, b.upc, b.title, b.description, b.price,
           b.star_rating, b.availability, b.url,
           c.name as category_name
    FROM books b
    JOIN categories c ON b.category_id = c.id
"""


# ─── Add-on E: Vector similarity setup ───────────────────────────────────────
def init_vector_db():
    """Initialize ChromaDB with sentence-transformer embeddings."""
    global chroma_collection
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Delete and recreate to ensure fresh data
        try:
            client.delete_collection("books")
        except Exception:
            pass

        chroma_collection = client.get_or_create_collection(
            name="books",
            embedding_function=embedding_fn,
        )

        # Load books and add to collection
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(BOOK_QUERY)
        rows = cursor.fetchall()
        conn.close()

        if rows and chroma_collection.count() == 0:
            ids = [str(row["id"]) for row in rows]
            documents = [
                f"{row['title']} {row['description'] or ''}" for row in rows
            ]
            metadatas = [
                {"title": row["title"], "category": row["category_name"], "book_id": row["id"]}
                for row in rows
            ]
            chroma_collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"  Vector DB: indexed {len(ids)} books")

    except Exception as e:
        print(f"  Warning: Vector DB init failed: {e}")
        chroma_collection = None


# ─── Add-on F: Knowledge graph setup ─────────────────────────────────────────
def init_knowledge_graph():
    """Build a NetworkX knowledge graph of Category—contains—Book relationships."""
    global knowledge_graph
    try:
        G = nx.Graph()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(BOOK_QUERY)
        rows = cursor.fetchall()
        conn.close()

        # Add category and book nodes + edges
        for row in rows:
            cat_node = f"category:{row['category_name']}"
            book_node = f"book:{row['id']}"

            G.add_node(cat_node, type="category", name=row["category_name"])
            G.add_node(book_node, type="book", id=row["id"], title=row["title"],
                       category=row["category_name"])
            G.add_edge(cat_node, book_node, relation="contains")

        # Add similar_to edges from vector similarity (if available)
        if chroma_collection is not None:
            for row in rows:
                try:
                    results = chroma_collection.query(
                        query_texts=[f"{row['title']} {row['description'] or ''}"],
                        n_results=4  # top 3 similar + self
                    )
                    if results and results["ids"] and results["ids"][0]:
                        for sim_id in results["ids"][0]:
                            if sim_id != str(row["id"]):
                                book_a = f"book:{row['id']}"
                                book_b = f"book:{sim_id}"
                                if not G.has_edge(book_a, book_b):
                                    G.add_edge(book_a, book_b, relation="similar_to")
                except Exception:
                    pass

        knowledge_graph = G
        print(f"  Knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    except Exception as e:
        print(f"  Warning: Knowledge graph init failed: {e}")
        knowledge_graph = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and resources at startup."""
    global classifier
    print("Starting Project SENTINEL API...")

    # Load classifier
    if os.path.exists(MODEL_FILE):
        classifier = joblib.load(MODEL_FILE)
        print(f"  Classifier loaded from {MODEL_FILE}")
    else:
        print(f"  Warning: {MODEL_FILE} not found. /classify will fail.")

    # Init vector DB (Add-on E)
    print("  Initializing vector DB (Add-on E)...")
    init_vector_db()

    # Init knowledge graph (Add-on F)
    print("  Building knowledge graph (Add-on F)...")
    init_knowledge_graph()

    print("  Ready!\n")
    yield


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Project SENTINEL API",
    description="Book data service with classification, vector similarity, and knowledge graph.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Core Endpoints ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/books", response_model=list[BookResponse])
def list_books(category: str | None = Query(None, description="Filter by category name"), limit=0):
    """List all books, optionally filtered by category."""
    conn = get_db()
    cursor = conn.cursor()


    if category:
        cursor.execute(BOOK_QUERY + " WHERE c.name = ?", (category,))
    else:
        cursor.execute(BOOK_QUERY)

    rows = cursor.fetchall()
    rows = rows[:limit] if limit else rows
    conn.close()

    return [row_to_book(row) for row in rows]


@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: int):
    """Get a single book by ID. Returns 404 if not found."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(BOOK_QUERY + " WHERE b.id = ?", (book_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")

    return row_to_book(row)


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Predict category from title and description."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier model not loaded")

    text = f"{request.title} {request.description}"
    predicted = classifier.predict([text])[0]

    # Get confidence (max probability)
    probabilities = classifier.predict_proba([text])[0]
    confidence = float(max(probabilities))

    return {
        "predicted_category": predicted,
        "confidence": round(confidence, 4),
    }


# ─── Add-on E: Vector Similarity ─────────────────────────────────────────────
@app.get("/similar/{book_id}", response_model=list[SimilarBookResponse])
def get_similar(book_id: int):
    """Get 5 most similar books by vector similarity (Add-on E)."""
    if chroma_collection is None:
        raise HTTPException(status_code=503, detail="Vector similarity not available")

    try:
        # Get the source book's text
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(BOOK_QUERY + " WHERE b.id = ?", (book_id,))
        source = cursor.fetchone()
        conn.close()

        if not source:
            raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")

        query_text = f"{source['title']} {source['description'] or ''}"

        # Query ChromaDB for similar books (6 = 5 similar + self)
        results = chroma_collection.query(query_texts=[query_text], n_results=6)

        similar = []
        if results and results["ids"] and results["ids"][0]:
            for i, str_id in enumerate(results["ids"][0]):
                if str_id != str(book_id):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity = max(0.0, 1.0 - distance / 10.0)
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    similar.append({
                        "id": int(str_id),
                        "title": metadata.get("title", ""),
                        "category": metadata.get("category", ""),
                        "similarity_score": round(similarity, 4),
                    })

        return similar[:5]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


# ─── Add-on F: Knowledge Graph ───────────────────────────────────────────────
@app.get("/graph/book/{book_id}", response_model=GraphNeighbourhood)
def get_graph_neighbourhood(book_id: int):
    """Get a book's neighbourhood in the knowledge graph (Add-on F)."""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    # Get the book from DB
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(BOOK_QUERY + " WHERE b.id = ?", (book_id,))
    book_row = cursor.fetchone()
    conn.close()

    if not book_row:
        raise HTTPException(status_code=404, detail=f"Book with id {book_id} not found")

    book_node = f"book:{book_id}"
    if book_node not in knowledge_graph:
        raise HTTPException(status_code=404, detail=f"Book {book_id} not in knowledge graph")

    book = row_to_book(book_row)

    # Find same-category books
    same_category = []
    similar_books = []

    for neighbor in knowledge_graph.neighbors(book_node):
        edge_data = knowledge_graph.edges[book_node, neighbor]
        relation = edge_data.get("relation", "")

        if relation == "contains":
            # This is the category node — find other books in same category
            for cat_neighbor in knowledge_graph.neighbors(neighbor):
                if cat_neighbor != book_node and cat_neighbor.startswith("book:"):
                    node_data = knowledge_graph.nodes[cat_neighbor]
                    same_category.append({
                        "id": node_data.get("id"),
                        "title": node_data.get("title", ""),
                        "category": node_data.get("category", ""),
                    })
        elif relation == "similar_to" and neighbor.startswith("book:"):
            node_data = knowledge_graph.nodes[neighbor]
            similar_books.append({
                "id": node_data.get("id"),
                "title": node_data.get("title", ""),
                "category": node_data.get("category", ""),
            })

    return {
        "book": book,
        "same_category": same_category[:10],  # limit to 10
        "similar_books": similar_books[:5],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
