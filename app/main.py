from flask import Flask, request, jsonify, render_template
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
import torch
import faiss
from typing import Dict, Any, Optional
import os
from threading import Lock
from contextlib import contextmanager
from app.utils import process_search_results

from app.logger_config import setup_logging, query_stats
import logging
import time

# Initialize logging
setup_logging()
logging.info("Starting FAISS Search API")


class FAISSManager:
    """Singleton class to manage FAISS index and GPU resources"""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FAISSManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.index: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        self.gpu_resources: Optional[Any] = None
        self.index_path = os.getenv('FAISS_INDEX_PATH', '../app_data/faiss_index')
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        self.gpu_lock = Lock()
        self._initialized = True

    @contextmanager
    def gpu_context(self):
        """Context manager for thread-safe GPU operations"""
        try:
            self.gpu_lock.acquire()
            yield
        finally:
            self.gpu_lock.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def initialize(self):
        """Initialize FAISS index with GPU support"""
        try:
            print("Checking GPU availability...")
            if torch.cuda.is_available():
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()

                # Configure GPU resources
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_resources.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            else:
                print("No GPU found, using CPU")

            print("Initializing embedding model...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/multi-qa-distilbert-cos-v1",
                encode_kwargs={
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'batch_size': self.batch_size
                }
            )

            print(f"Loading FAISS index from {self.index_path}...")
            self.index = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )

            if torch.cuda.is_available() and self.gpu_resources is not None:
                print("Moving index to GPU...")
                with self.gpu_context():
                    gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index.index)
                    self.index.index = gpu_index
                print("Successfully moved FAISS index to GPU")

            print("FAISS initialization completed successfully")
            return True

        except Exception as e:
            print(f"Error initializing FAISS: {e}")
            raise

    def search(self, query: str, k: int) -> Any:
        """Thread-safe search operation"""
        with self.gpu_context():
            return self.index.similarity_search_with_score(query, k=k, fetch_k=k * 2)

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_mb': torch.cuda.memory_allocated(0) / 1024 ** 2,
            'reserved_mb': torch.cuda.memory_reserved(0) / 1024 ** 2,
            'max_mb': torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        }


# Initialize FAISS manager as a global singleton
faiss_manager = FAISSManager()
faiss_manager.initialize()


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)

    @app.route('/')
    def welcome():
        """Welcome page"""
        try:
            gpu_available = torch.cuda.is_available()
            gpu_device = torch.cuda.get_device_name(0) if gpu_available else "Not Available"
            index_status = "Loaded" if faiss_manager.index is not None else "Not Loaded"

            return render_template(
                'hello.html',
                gpu_status="Available ✅" if gpu_available else "Not Available ❌",
                gpu_device=gpu_device,
                index_status=index_status
            )
        except Exception as e:
            print(f"Error in welcome route: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/search', methods=['POST'])
    def search() -> Dict[str, Any]:
        """Search endpoint with error handling, input validation, and stats tracking"""
        start_time = time.time()
        query = ""

        try:
            data = request.get_json()
            if not data or 'query' not in data:
                logging.warning("Search request received with no query")
                return jsonify({
                    'status': 'error',
                    'message': 'No query provided'
                }), 400

            query = data['query'].strip()
            if not query:
                logging.warning("Search request received with empty query")
                return jsonify({
                    'status': 'error',
                    'message': 'Empty query'
                }), 400

            k = min(int(data.get('k', 200)), 1000)  # Limit maximum results

            logging.info(f"Processing search query: '{query}' with k={k}")
            results = faiss_manager.search(query, k)
            processed_results = process_search_results(results)

            time_taken = time.time() - start_time
            query_stats.log_query(query, True, time_taken)

            logging.info(f"Search completed in {time_taken:.2f}s with {len(processed_results)} results")

            return jsonify({
                'status': 'success',
                'query': query,
                'results_count': len(processed_results),
                'results': processed_results,
                'time_taken': round(time_taken, 3)
            })

        except ValueError as e:
            time_taken = time.time() - start_time
            query_stats.log_query(query, False, time_taken)
            logging.error(f"Invalid input error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Invalid input: {str(e)}'
            }), 400
        except Exception as e:
            time_taken = time.time() - start_time
            query_stats.log_query(query, False, time_taken)
            logging.error(f"Search error: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'Internal server error'
            }), 500

    @app.route('/health', methods=['GET'])
    def health_check() -> Dict[str, Any]:
        """Enhanced health check endpoint with memory stats"""
        try:
            gpu_stats = faiss_manager.get_gpu_stats()
            return jsonify({
                'status': 'healthy',
                'gpu_available': torch.cuda.is_available(),
                'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'index_loaded': faiss_manager.index is not None,
                'gpu_memory': gpu_stats,
                'batch_size': faiss_manager.batch_size
            })
        except Exception as e:
            print(f"Health check error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Global exception handler"""
        print(f"Unhandled exception: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

    @app.route('/stats', methods=['GET'])
    def get_stats() -> Dict[str, Any]:
        """Get query statistics"""
        try:
            stats = query_stats._load_stats()
            return jsonify({
                'status': 'success',
                'stats': {
                    'total_queries': stats['total_queries'],
                    'successful_queries': stats['successful_queries'],
                    'failed_queries': stats['failed_queries'],
                    'avg_time': round(stats['avg_time'], 3),
                    'recent_queries': stats['queries'][-10:]  # Last 10 queries
                }
            })
        except Exception as e:
            logging.error(f"Error getting stats: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'Error retrieving statistics'
            }), 500

    return app


# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
