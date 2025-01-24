import requests
import time
import asyncio
import aiohttp
from datetime import datetime
import statistics
from typing import List, Dict
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
                    "start_time": start_time,
                    "end_time": time.time(),
                    "duration": duration,
                    "status": status,
                    "success": 200 <= status < 300
                }

        except Exception as e:
            duration = time.time() - start_time
            return {
                "start_time": start_time,
                "end_time": time.time(),
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

        global_start_time = time.time()

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

        global_end_time = time.time()
        self.print_results(global_start_time, global_end_time, num_requests)

    def print_results(self, global_start_time: float, global_end_time: float, num_requests: int) -> None:
        """Print detailed performance metrics"""
        successful_requests = num_requests - self.failed_requests
        durations = [r["duration"] for r in self.results if r["success"]]

        if not durations:
            print("\nAll requests failed!")
            return

        total_time = global_end_time - global_start_time

        # Calculate concurrent execution metrics
        start_times = [r["start_time"] for r in self.results]
        end_times = [r["end_time"] for r in self.results]

        # Calculate overlap statistics
        time_points = []
        for start, end in zip(start_times, end_times):
            time_points.append((start, 1))  # 1 for start
            time_points.append((end, -1))  # -1 for end

        time_points.sort()

        current_concurrent = 0
        max_concurrent = 0
        concurrent_counts = []

        for _, change in time_points:
            current_concurrent += change
            concurrent_counts.append(current_concurrent)
            max_concurrent = max(max_concurrent, current_concurrent)

        avg_concurrent = statistics.mean(concurrent_counts) if concurrent_counts else 0

        print("\n=== Stress Test Results ===")
        print(f"\nOverall Performance:")
        print(f"├─ Total Wall Clock Time: {total_time:.2f} seconds")
        print(f"├─ Cumulative Processing Time: {sum(durations):.2f} seconds")
        print(f"├─ Average Concurrent Requests: {avg_concurrent:.1f}")
        print(f"└─ Max Concurrent Requests: {max_concurrent}")

        print(f"\nPer-Request Metrics:")
        print(f"├─ Average Response Time: {statistics.mean(durations):.3f} seconds")
        print(f"├─ Median Response Time: {statistics.median(durations):.3f} seconds")
        print(f"├─ Min Response Time: {min(durations):.3f} seconds")
        print(f"├─ Max Response Time: {max(durations):.3f} seconds")
        print(f"└─ Standard Deviation: {statistics.stdev(durations):.3f} seconds")

        print(f"\nThroughput Metrics:")
        print(f"├─ Total Requests: {num_requests}")
        print(f"├─ Successful Requests: {successful_requests}")
        print(f"├─ Failed Requests: {self.failed_requests}")
        print(f"├─ Success Rate: {(successful_requests / num_requests) * 100:.1f}%")
        print(f"└─ Requests/second: {num_requests / total_time:.2f}")

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
                    concurrent_limit: int = 1,
                    base_url: str = "http://localhost:5000",
                    query: str = "nitrosamine",
                    k: int = 50):
    """Helper function to run the stress test"""
    tester = APIStressTester(base_url)
    asyncio.run(tester.run_stress_test(
        num_requests=requests,
        concurrent_limit=concurrent_limit,
        query=query,
        k=k
    ))


if __name__ == "__main__":
    run_stress_test()