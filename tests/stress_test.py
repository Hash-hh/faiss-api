import requests
import time
import asyncio
import aiohttp
from datetime import datetime
import statistics
from typing import List, Dict
import concurrent.futures
from tqdm import tqdm


class APIStressTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.results: List[Dict] = []
        self.failed_requests = 0

    async def async_make_request(self, session: aiohttp.ClientSession, query: str, k: int) -> Dict:
        """Make a single async request"""
        url = f"{self.base_url}/search"
        payload = {"query": query, "k": k}

        start_time = time.time()
        try:
            async with session.post(url, json=payload) as response:
                duration = time.time() - start_time
                status = response.status

                return {
                    "duration": duration,
                    "status": status,
                    "success": 200 <= status < 300
                }

        except Exception as e:
            duration = time.time() - start_time
            return {
                "duration": duration,
                "status": 0,
                "success": False,
                "error": str(e)
            }

    async def run_stress_test(self,
                              num_requests: int = 50,
                              query: str = "nitrosamine",
                              k: int = 5,
                              concurrent_limit: int = 10) -> None:
        """Run stress test with concurrent requests"""

        print(f"\n=== Starting Stress Test ===")
        print(f"Total Requests: {num_requests}")
        print(f"Concurrency Limit: {concurrent_limit}")
        print(f"Query: '{query}'")

        start_time = time.time()

        # Create a connection pool with limits
        connector = aiohttp.TCPConnector(limit=concurrent_limit)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for _ in range(num_requests):
                task = self.async_make_request(session, query, k)
                tasks.append(task)

            # Show progress bar while waiting for results
            with tqdm(total=num_requests, desc="Requests") as pbar:
                for future in asyncio.as_completed(tasks):
                    result = await future
                    self.results.append(result)
                    if not result["success"]:
                        self.failed_requests += 1
                    pbar.update(1)

        total_time = time.time() - start_time
        self.print_results(total_time, num_requests)

    def print_results(self, total_time: float, num_requests: int) -> None:
        """Print detailed performance metrics"""
        successful_requests = num_requests - self.failed_requests
        durations = [r["duration"] for r in self.results if r["success"]]

        if not durations:
            print("\nAll requests failed!")
            return

        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        if len(durations) > 1:
            stdev_duration = statistics.stdev(durations)
        else:
            stdev_duration = 0

        requests_per_second = num_requests / total_time

        print("\n=== Stress Test Results ===")
        print(f"\nTime Metrics:")
        print(f"├─ Total Time: {total_time:.2f} seconds")
        print(f"├─ Average Response Time: {avg_duration:.3f} seconds")
        print(f"├─ Median Response Time: {median_duration:.3f} seconds")
        print(f"├─ Min Response Time: {min_duration:.3f} seconds")
        print(f"├─ Max Response Time: {max_duration:.3f} seconds")
        print(f"└─ Standard Deviation: {stdev_duration:.3f} seconds")

        print(f"\nRequest Metrics:")
        print(f"├─ Total Requests: {num_requests}")
        print(f"├─ Successful Requests: {successful_requests}")
        print(f"├─ Failed Requests: {self.failed_requests}")
        print(f"├─ Success Rate: {(successful_requests / num_requests) * 100:.1f}%")
        print(f"└─ Requests/second: {requests_per_second:.2f}")

        if self.failed_requests > 0:
            print(f"\nError Distribution:")
            error_counts = {}
            for result in self.results:
                if not result["success"]:
                    status = result.get("status", "Unknown")
                    error_counts[status] = error_counts.get(status, 0) + 1

            for status, count in error_counts.items():
                print(f"├─ Status {status}: {count} occurrences")


def run_stress_test(requests: int = 50,
                    concurrent_limit: int = 10,
                    base_url: str = "http://localhost:5000",
                    query: str = "nitrosamine",
                    k: int = 5):
    """Helper function to run the stress test"""
    tester = APIStressTester(base_url)

    # Run the async stress test
    asyncio.run(tester.run_stress_test(
        num_requests=requests,
        concurrent_limit=concurrent_limit,
        query=query,
        k=k
    ))


if __name__ == "__main__":
    # Configuration
    NUM_REQUESTS = 50
    CONCURRENT_LIMIT = 10  # Maximum concurrent connections
    BASE_URL = "http://localhost:5000"

    print("Starting stress test...")
    run_stress_test(
        requests=NUM_REQUESTS,
        concurrent_limit=CONCURRENT_LIMIT,
        base_url=BASE_URL
    )