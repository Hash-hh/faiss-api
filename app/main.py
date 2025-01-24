from flask import Flask, request, jsonify, render_template
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import faiss
from typing import Dict, Any, Optional
import os
from app.utils import process_search_results


# Global variables
FAISS_INDEX: Optional[Any] = None
EMBEDDING_MODEL: Optional[Any] = None
GPU_RESOURCES: Optional[Any] = None
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', '../app_data/faiss_index')
DEFAULT_RESULTS = 200

def init_faiss() -> None:
    """Initialize FAISS index with GPU support"""
    global FAISS_INDEX, EMBEDDING_MODEL, GPU_RESOURCES

    try:
        print("Checking GPU availability...")
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_device(0)
            GPU_RESOURCES = faiss.StandardGpuResources()
        else:
            print("No GPU found, using CPU")

        print("Initializing embedding model...")
        EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-distilbert-cos-v1",
            encode_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        FAISS_INDEX = FAISS.load_local(
            FAISS_INDEX_PATH,
            EMBEDDING_MODEL,
            allow_dangerous_deserialization=True
        )

        if torch.cuda.is_available() and GPU_RESOURCES is not None:
            print("Moving index to GPU...")
            gpu_index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, FAISS_INDEX.index)
            FAISS_INDEX.index = gpu_index
            print("Successfully moved FAISS index to GPU")

        print("FAISS initialization completed successfully")

    except Exception as e:
        print(f"Error initializing FAISS: {e}")
        raise

# Initialize FAISS when the module is loaded
init_faiss()

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)

    @app.route('/')
    def welcome():
        """Welcome page"""
        gpu_available = torch.cuda.is_available()
        gpu_device = torch.cuda.get_device_name(0) if gpu_available else "Not Available"
        index_status = "Loaded" if FAISS_INDEX is not None else "Not Loaded"

        return render_template(
            'hello.html',  # Move hello.html to templates folder
            gpu_status="Available ✅" if gpu_available else "Not Available ❌",
            gpu_device=gpu_device,
            index_status=index_status
        )

    @app.route('/search', methods=['POST'])
    def search() -> Dict[str, Any]:
        try:
            data = request.get_json()

            if not data or 'query' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'No query provided'
                }), 400

            query = data['query']
            k = int(data.get('k', DEFAULT_RESULTS))

            if not query.strip():
                return jsonify({
                    'status': 'error',
                    'message': 'Empty query'
                }), 400

            results = FAISS_INDEX.similarity_search_with_score(
                query,
                k=k,
                fetch_k=k * 2
            )

            processed_results = process_search_results(results)

            return jsonify({
                'status': 'success',
                'query': query,
                'results_count': len(processed_results),
                'results': processed_results
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/health', methods=['GET'])
    def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'gpu_available': torch.cuda.is_available(),
            'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'index_loaded': FAISS_INDEX is not None
        })

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)