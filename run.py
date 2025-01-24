import os
from app.main import create_app

if __name__ == '__main__':
    # Set the path to your FAISS index
    os.environ['FAISS_INDEX_PATH'] = r'C:\Users\hasha\PycharmProjects\faiss-api\app_data\faiss_index'  # only for hash's windows

    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
    