# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
# Allow specific origins for local and hosted environments
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
CORS(app, resources={r"/rag": {"origins": ALLOWED_ORIGINS}})

# Load RAG components
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('corpus_index.faiss')
    with open('corpus_data.json', 'r') as f:
        corpus = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure corpus_index.faiss and corpus_data.json are in the same directory as app.py.")
    raise

# Configure Gemini API
GEMINI_API_KEY = os.getenv('AIzaSyDZlTkys2Kpfah0Ki8N1CgAt-aqwierAPc')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

@app.route('/rag', methods=['POST'])
def rag():
    """Handle RAG queries with fallback to Gemini 2.0 Flash."""
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # RAG retrieval
        query_embedding = model.encode([query])[0]
        D, I = index.search(np.array([query_embedding]), k=5)

        # Check relevance
        if all(d > 1.0 for d in D[0]):
            try:
                response = genai.generate_content(
                    model="gemini-2.0-flash",
                    prompt=query,
                    generation_config={"temperature": 0.7}
                )
                return jsonify({'response': response.text or 'Sorry, I couldn’t generate a relevant answer. Try a business-related question!'})
            except Exception as gemini_error:
                return jsonify({'response': 'Sorry, I don’t have information on that. Try a business-related question!'})
        else:
            response = [corpus[i] for i in I[0]]
            return jsonify({'response': ' '.join(response)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (for hosted platforms) or default to 5000 (local)
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV', 'development') == 'development')
