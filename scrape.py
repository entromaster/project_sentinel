"""
Core A — Web Scraper (Browser Automation with Playwright)

Drives a real Chromium browser to crawl https://books.toscrape.com,
collecting ~60-100 books across 5 categories with pagination support.
Saves results to data/books.json.
"""

import json
import os
import time
import re
from playwright.sync_api import sync_playwright

BASE_URL = "https://books.toscrape.com"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "books.json")

# Number of categories to scrape
NUM_CATEGORIES = 5

# Star rating text to number mapping
STAR_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def parse_star_rating(classes: list[str]) -> int:
    """Extract numeric star rating from CSS class list like ['star-rating', 'Three']."""
    for cls in classes:
        lower = cls.lower()
        if lower in STAR_MAP:
            return STAR_MAP[lower]
    return 0


def scrape_book_detail(page, book_url: str) -> dict:
    """Visit a book's detail page and extract full info."""
    page.goto(book_url, wait_until="domcontentloaded")
    time.sleep(0.3)  # polite delay

    title = page.locator("h1").first.inner_text()

    # Description — sometimes missing
    desc_el = page.locator("#product_description ~ p")
    description = desc_el.inner_text() if desc_el.count() > 0 else ""

    # Product info table
    info_rows = page.locator("table.table-striped tr")
    product_info = {}
    for i in range(info_rows.count()):
        row = info_rows.nth(i)
        th = row.locator("th").inner_text().strip()
        td = row.locator("td").inner_text().strip()
        product_info[th] = td

    upc = product_info.get("UPC", "")
    price_text = product_info.get("Price (incl. tax)", "£0.00")
    price = float(price_text.replace("£", "").replace(",", ""))
    availability = product_info.get("Availability", "")

    # Star rating — scope to product_main to avoid matching recommended books
    star_el = page.locator("div.product_main p.star-rating")
    star_classes = star_el.get_attribute("class").split() if star_el.count() > 0 else []
    star_rating = parse_star_rating(star_classes)

    return {
        "title": title,
        "description": description,
        "price": price,
        "star_rating": star_rating,
        "availability": availability,
        "upc": upc,
        "url": book_url,
    }


def scrape_category(page, category_name: str, category_url: str) -> list[dict]:
    """Scrape all books from a category, handling pagination."""
    books = []
    current_url = category_url

    while current_url:
        page.goto(current_url, wait_until="domcontentloaded")
        time.sleep(0.5)  # polite delay

        # Get all book links on this page
        book_links = page.locator("h3 a")
        book_urls = []
        for i in range(book_links.count()):
            href = book_links.nth(i).get_attribute("href")
            # Resolve relative URL
            if href.startswith("../../.."):
                book_url = BASE_URL + "/catalogue/" + href.replace("../../../", "")
            elif href.startswith("../"):
                book_url = BASE_URL + "/catalogue/" + href.replace("../", "")
            elif not href.startswith("http"):
                # Handle relative URLs within category pagination
                base_path = current_url.rsplit("/", 1)[0]
                book_url = base_path + "/" + href
                # Normalize: if this results in ../../.. style, fix it
                if "../../.." in book_url:
                    book_url = BASE_URL + "/catalogue/" + href.replace("../../../", "")
            else:
                book_url = href
            book_urls.append(book_url)

        print(f"  Found {len(book_urls)} books on page")

        # Visit each book detail page
        for url in book_urls:
            try:
                book = scrape_book_detail(page, url)
                book["category"] = category_name
                books.append(book)
                print(f"    Scraped: {book['title'][:60]}")
            except Exception as e:
                print(f"    Error scraping {url}: {e}")

        # Check for "next" button for pagination
        page.goto(current_url, wait_until="domcontentloaded")
        time.sleep(0.3)
        next_btn = page.locator("li.next a")
        if next_btn.count() > 0:
            next_href = next_btn.get_attribute("href")
            base_path = current_url.rsplit("/", 1)[0]
            current_url = base_path + "/" + next_href
            print(f"  -> Following pagination to next page")
        else:
            current_url = None  # No more pages

    return books


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Project SENTINEL — Web Scraper")
    print("Target: https://books.toscrape.com")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to home page
        print("\n[1] Loading homepage...")
        page.goto(BASE_URL, wait_until="domcontentloaded")
        time.sleep(1)

        # Discover categories from sidebar
        print("[2] Discovering categories...")
        category_links = page.locator("div.side_categories ul li ul li a")
        categories = []
        for i in range(category_links.count()):
            name = category_links.nth(i).inner_text().strip()
            href = category_links.nth(i).get_attribute("href")
            full_url = BASE_URL + "/" + href
            categories.append((name, full_url))

        print(f"    Found {len(categories)} categories total")

        # Select first NUM_CATEGORIES
        selected = categories[:NUM_CATEGORIES]
        print(f"    Scraping {len(selected)} categories: {[c[0] for c in selected]}")

        # Scrape each category
        all_books = []
        for cat_name, cat_url in selected:
            print(f"\n[Scraping] Category: {cat_name}")
            cat_books = scrape_category(page, cat_name, cat_url)
            all_books.extend(cat_books)
            print(f"  -> {len(cat_books)} books collected from {cat_name}")

        browser.close()

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_books, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"DONE — {len(all_books)} books scraped across {len(selected)} categories")
    print(f"Saved to {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    # Quick summary
    from collections import Counter
    cat_counts = Counter(b["category"] for b in all_books)
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} books")


if __name__ == "__main__":
    main()
