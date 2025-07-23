# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app, resources={r"/rag": {"origins": ["http://localhost:3000", "http://your-ec2-public-ip"]}})

# Load RAG components
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('corpus_index.faiss')
with open('corpus_data.json', 'r') as f:
    corpus = json.load(f)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')) # Replace or use env var

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
    app.run()