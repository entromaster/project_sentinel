"""
Core B — Database Loader (Idempotent SQLite)

Loads scraped book data from data/books.json into a SQLite database (books.db).
Uses UPC as a natural key to ensure idempotency — running twice won't duplicate rows.
"""

import json
import os
import sqlite3

DATA_FILE = os.path.join("data", "books.json")
DB_FILE = "books.db"


def create_schema(conn: sqlite3.Connection):
    """Create the database schema (categories + books tables)."""
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upc TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            description TEXT,
            price REAL,
            star_rating INTEGER,
            availability TEXT,
            url TEXT,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );
    """)
    conn.commit()


def load_data(conn: sqlite3.Connection, books: list[dict]):
    """Load books into the database idempotently using INSERT OR IGNORE on UPC."""
    cursor = conn.cursor()

    categories_inserted = 0
    books_inserted = 0
    books_skipped = 0

    for book in books:
        category_name = book["category"]

        # Insert category (ignore if already exists)
        cursor.execute(
            "INSERT OR IGNORE INTO categories (name) VALUES (?)",
            (category_name,)
        )

        # Get category id
        cursor.execute(
            "SELECT id FROM categories WHERE name = ?",
            (category_name,)
        )
        category_id = cursor.fetchone()[0]

        # Insert book (ignore if UPC already exists — idempotency)
        cursor.execute(
            """INSERT OR IGNORE INTO books
               (upc, title, category_id, description, price, star_rating, availability, url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                book["upc"],
                book["title"],
                category_id,
                book.get("description", ""),
                book.get("price", 0.0),
                book.get("star_rating", 0),
                book.get("availability", ""),
                book.get("url", ""),
            )
        )

        if cursor.rowcount > 0:
            books_inserted += 1
        else:
            books_skipped += 1

    conn.commit()
    return books_inserted, books_skipped


def main():
    print("=" * 60)
    print("Project SENTINEL — Database Loader")
    print("=" * 60)

    # Load scraped data
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run scrape.py first.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        books = json.load(f)

    print(f"Loaded {len(books)} books from {DATA_FILE}")

    # Connect to SQLite
    conn = sqlite3.connect(DB_FILE)

    # Create schema
    print("Creating schema...")
    create_schema(conn)

    # Load data
    print("Loading data (idempotent — safe to run multiple times)...")
    inserted, skipped = load_data(conn, books)

    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM books")
    total_books = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM categories")
    total_categories = cursor.fetchone()[0]

    conn.close()

    print(f"\nResults:")
    print(f"  Books inserted:  {inserted}")
    print(f"  Books skipped:   {skipped} (already in DB — idempotency working)")
    print(f"  Total in DB:     {total_books} books, {total_categories} categories")
    print(f"  Database file:   {DB_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
