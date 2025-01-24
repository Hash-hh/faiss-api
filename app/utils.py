import numpy as np


def euclidean_to_cosine_similarity(euclidean_distance):
    """Convert Euclidean distance to Cosine similarity"""
    return 1 - (euclidean_distance ** 2) / 2


def cosine_to_confidence(cosine_similarity):
    """Convert Cosine similarity to confidence percentage"""
    return round(((cosine_similarity + 1) / 2) * 100, 2)


def process_search_results(results):
    """Process FAISS search results into JSON-serializable format"""
    try:
        return [
            {
                "chunk": result[0].page_content,
                "metadata": {
                    "file_name": result[0].metadata.get("file_name", "Unknown"),
                    "file_path": result[0].metadata.get("file_path", "Unknown"),
                    "page_number": result[0].metadata.get("page_number", "Unknown"),
                    "date_modified": result[0].metadata.get("date_modified", "Unknown"),
                },
                "score": float(result[1]),  # Convert numpy float to Python float
                "confidence": cosine_to_confidence(euclidean_to_cosine_similarity(float(result[1]))),
                "cosine_similarity": float(euclidean_to_cosine_similarity(float(result[1])))
            }
            for result in results
        ]
    except Exception as e:
        print(f"Error processing results: {e}")
        return []