import requests
import json
from datetime import datetime
import time
from typing import Dict, Any
import sys


class APITester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()  # Use session for better performance
        self.last_response_time = None

    def _make_request(self, method: str, endpoint: str, data: Dict = None, retries: int = 3) -> requests.Response:
        """Make HTTP request with retry logic and timing"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        attempt = 0

        while attempt < retries:
            try:
                start_time = time.time()
                response = self.session.request(method, url, json=data)
                self.last_response_time = time.time() - start_time

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt == retries:
                    raise
                print(f"Attempt {attempt} failed: {str(e)}. Retrying...")
                time.sleep(1)  # Wait 1 second before retry

    def test_search_endpoint(self, query: str = "nitrosamine", k: int = 5) -> Dict[str, Any]:
        """Test the search API with given query"""
        print(f"\n=== Testing Search Endpoint ===")
        print(f"Query: '{query}' (k={k})")

        try:
            # Send POST request
            response = self._make_request(
                'POST',
                'search',
                data={"query": query, "k": k}
            )

            data = response.json()

            # Print performance metrics
            print(f"\nPerformance Metrics:")
            print(f"Response Time: {self.last_response_time:.3f} seconds")
            print(f"Response Size: {len(response.content):,} bytes")

            # Print results summary
            print(f"\nResults Summary:")
            print(f"Status: {data['status']}")
            print(f"Total Results: {data['results_count']}")

            # Print detailed results
            print("\nTop Results:")
            for idx, result in enumerate(data['results'][:k], 1):
                print(f"\nResult {idx}:")
                print("=" * 50)
                print(f"File: {result['metadata'].get('file_name', 'N/A')}")
                print(f"Page: {result['metadata'].get('page_number', 'N/A')}")
                print(f"Confidence: {result.get('confidence', 'N/A')}%")

                # Print chunk with better formatting
                chunk = result.get('chunk', '')
                if chunk:
                    print("\nSnippet:")
                    print("-" * 50)
                    print(f"{chunk[:300]}...")
                    print("-" * 50)

            # Save results to file
            self._save_results(data, query)

            return data

        except requests.exceptions.ConnectionError:
            print(f"ERROR: Could not connect to server at {self.base_url}")
            print("Make sure the server is running and the URL is correct.")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            print("ERROR: Server returned invalid JSON response")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

        return None

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        print("\n=== Testing Health Endpoint ===")

        try:
            response = self._make_request('GET', 'health')
            data = response.json()

            print(f"Response Time: {self.last_response_time:.3f} seconds")
            print("\nHealth Status:")
            print(f"├─ Status: {data['status']}")
            print(f"├─ GPU Available: {data['gpu_available']}")
            print(f"├─ GPU Device: {data['gpu_device']}")
            print(f"└─ Index Loaded: {data['index_loaded']}")

            return data

        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return None

    def _save_results(self, data: Dict, query: str) -> None:
        """Save results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{query.replace(' ', '_')}_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'query': query,
                        'response_time': self.last_response_time,
                    },
                    'data': data
                }, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


def main():
    # Configure the tester
    tester = APITester()

    # Test queries
    queries = [
        ("nitrosamine", 5),
        ("nitrosamine formation", 3),
        # Add more test queries as needed
    ]

    # Run health check first
    health_data = tester.test_health_endpoint()
    if not health_data or health_data.get('status') != 'healthy':
        print("\nWARNING: Server might not be healthy!")
        if input("Continue with tests? (y/n): ").lower() != 'y':
            sys.exit(1)

    # Run search tests
    for query, k in queries:
        tester.test_search_endpoint(query, k)
        time.sleep(1)  # Small delay between requests


if __name__ == "__main__":
    main()