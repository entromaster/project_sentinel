import time
import requests
import json

def wait_for_server():
    print("Waiting for server to start...")
    for _ in range(30):
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                print("Server is up!")
                return True
        except:
            pass
        time.sleep(2)
    return False

def run_tests():
    if not wait_for_server():
        print("Server did not start in time.")
        return

    results = []
    
    # Query 1: Health
    print("Running Query 1: /health")
    results.append("=== Query 1: GET /health ===")
    r = requests.get("http://localhost:8000/health")
    results.append(f"Status Code: {r.status_code}")
    results.append(json.dumps(r.json(), indent=2))
    results.append("\n")

    # Query 2: Get Books by Category
    print("Running Query 2: /books?category=Mystery")
    results.append("=== Query 2: GET /books?category=Mystery ===")
    r = requests.get("http://localhost:8000/books?category=Mystery")
    results.append(f"Status Code: {r.status_code}")
    books = r.json()
    results.append(f"Found {len(books)} books. First 2 books:")
    results.append(json.dumps(books[:2], indent=2))
    results.append("\n")

    # Query 3: Classify
    print("Running Query 3: /classify")
    results.append("=== Query 3: POST /classify ===")
    payload = {
        "title": "A Walk in the Woods",
        "description": "A fascinating journey through the Appalachian trail exploring nature and hiking."
    }
    r = requests.post("http://localhost:8000/classify", json=payload)
    results.append(f"Status Code: {r.status_code}")
    results.append(f"Payload: {json.dumps(payload)}")
    results.append("Response:")
    results.append(json.dumps(r.json(), indent=2))
    results.append("\n")

    # Query 4: Similar Books
    print("Running Query 4: /similar/2")
    results.append("=== Query 4: GET /similar/2 ===")
    r = requests.get("http://localhost:8000/similar/2")
    results.append(f"Status Code: {r.status_code}")
    results.append("Response:")
    results.append(json.dumps(r.json(), indent=2))
    results.append("\n")

    # Query 5: Knowledge Graph
    print("Running Query 5: /graph/book/2")
    results.append("=== Query 5: GET /graph/book/2 ===")
    r = requests.get("http://localhost:8000/graph/book/2")
    results.append(f"Status Code: {r.status_code}")
    results.append("Response:")
    # truncate same_category to avoid huge output
    graph_data = r.json()
    if 'same_category' in graph_data and len(graph_data['same_category']) > 3:
        graph_data['same_category'] = graph_data['same_category'][:3] + [{"note": "... truncated for brevity ..."}]
    results.append(json.dumps(graph_data, indent=2))
    results.append("\n")

    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    
    print("Saved results to test_results.txt")

if __name__ == "__main__":
    run_tests()
