import requests
import json
from datetime import datetime


def test_search_endpoint():
    """Test the search API with 'nitrosamine' query"""

    # API endpoint
    url = "http://localhost:5000/search"

    # Test query
    payload = {
        "query": "nitrosamine",
        "k": 5  # Limiting to 5 results for test
    }

    try:
        # Send POST request
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)

        # Check if request was successful
        response.raise_for_status()

        # Get the response data
        data = response.json()

        # Print results in a formatted way
        print("\n=== Search Results ===")
        print(f"Status: {data['status']}")
        print(f"Query: {data['query']}")
        print(f"Total Results: {data['results_count']}")
        print("\nTop Results:")

        for idx, result in enumerate(data['results'], 1):
            print(f"\nResult {idx}:")
            print(f"File: {result['metadata']['file_name']}")
            print(f"Page: {result['metadata']['page_number']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Snippet: {result['chunk'][:200]}...")  # First 200 chars of the chunk

        # Save full results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to: {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Error: {e}")


def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:5000/health")
        response.raise_for_status()
        data = response.json()

        print("\n=== Health Check ===")
        print(f"Status: {data['status']}")
        print(f"GPU Available: {data['gpu_available']}")
        print(f"GPU Device: {data['gpu_device']}")
        print(f"Index Loaded: {data['index_loaded']}")

    except requests.exceptions.RequestException as e:
        print(f"Error checking health: {e}")


if __name__ == "__main__":
    # Run tests
    print("Testing health endpoint...")
    test_health_endpoint()

    print("\nTesting search endpoint...")
    test_search_endpoint()