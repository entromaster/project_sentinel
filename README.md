# Project SENTINEL

A complete data pipeline: **scrape** books from the web → **load** into SQLite → **train** a category classifier → **serve** via FastAPI with vector similarity and knowledge graph add-ons.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Scrape books (drives a real browser)
python scrape.py

# 3. Load into SQLite (idempotent — safe to run multiple times)
python load_db.py

# 4. Train the classifier
python train.py

# 5. Start the API server
python app.py
# → Open http://localhost:8000/docs for Swagger UI
```

## Architecture

```
scrape.py → data/books.json → load_db.py → books.db → train.py → models/classifier.pkl
                                                    ↘
                                                  app.py (FastAPI)
                                                    ├── /health
                                                    ├── /books, /books/{id}
                                                    ├── /classify
                                                    ├── /similar/{id}    (Add-on E)
                                                    └── /graph/book/{id} (Add-on F)
```

## What Works

| Component | Status | Details |
|-----------|--------|---------|
| **Core A — Scraper** | ✅ | Playwright browser automation, 5 categories, pagination |
| **Core B — DB Loader** | ✅ | SQLite with idempotent loading (UPC natural key) |
| **Core C — Classifier** | ✅ | TF-IDF + LogisticRegression, stratified split, F1 report |
| **Core D — FastAPI** | ✅ | All 4 endpoints + Swagger docs at /docs |
| **Add-on E — Vector Similarity** | ✅ | sentence-transformers + ChromaDB, /similar/{id} |
| **Add-on F — Knowledge Graph** | ✅ | NetworkX graph, /graph/book/{id} |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check → `{"status": "ok"}` |
| GET | `/books` | List books, optional `?category=` filter |
| GET | `/books/{id}` | Single book by ID (404 if missing) |
| POST | `/classify` | Predict category from `{title, description}` |
| GET | `/similar/{id}` | 5 most similar books (vector similarity) |
| GET | `/graph/book/{id}` | Book's neighbourhood (category + similar) |

## Key Decisions

- **Playwright** over Selenium: modern API, built-in auto-waits, easier setup.
- **SQLite**: zero-config, file-based, perfect for this scale. `INSERT OR IGNORE` on UPC for idempotency.
- **Scikit-learn pipeline**: `TfidfVectorizer` → `LogisticRegression` as specified. `class_weight="balanced"` to handle category imbalance.
- **ChromaDB**: persistent file-based vector store. Uses `all-MiniLM-L6-v2` embeddings.
- **NetworkX**: lightweight in-memory graph, built at API startup.

## AI & Fallback Data Usage

- **AI Usage:** AI (Gemini) was used to generate the initial code structure, help with boilerplate, and review implementation decisions.
- **Fallback Data:** The provided fallback data was **not used**. All data was scraped live via `scrape.py`.

## What I'd Do With More Time

- Add proper logging instead of print statements.
- Write pytest test cases for each endpoint.
- Add a Dockerfile for reproducible deployment.
- Scrape all 50 categories for a richer dataset and better classifier performance.
- Add caching and connection pooling for the SQLite reads.
- Enrich book data from Open Library API (author info, cover images).
