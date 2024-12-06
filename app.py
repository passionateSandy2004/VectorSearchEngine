from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage for user embeddings
user_data = {}

def make_embedding(user_name, string_list):
    """
    Takes a username and a list of strings, computes embeddings, 
    and stores them in the global user_data dictionary.
    """
    embeddings = model.encode(string_list, convert_to_tensor=True)
    user_data[user_name] = {
        "string_list": string_list,
        "embeddings": embeddings
    }

@app.route('/store', methods=['POST'])
def store_strings():
    """
    API endpoint to store a list of strings for a user.
    Expects JSON payload with 'user_name' and 'string_list'.
    """
    data = request.json
    user_name = data.get("user_name")
    string_list = data.get("string_list")
    
    if not user_name or not string_list:
        return jsonify({"error": "user_name and string_list are required"}), 400
    
    make_embedding(user_name, string_list)
    return jsonify({"message": f"Strings stored successfully for user: {user_name}"}), 200

@app.route('/query', methods=['POST'])
def query_string():
    """
    API endpoint to query the most similar string for a user.
    Expects JSON payload with 'user_name' and 'query_string'.
    """
    data = request.json
    user_name = data.get("user_name")
    query_string = data.get("query_string")
    
    if not user_name or not query_string:
        return jsonify({"error": "user_name and query_string are required"}), 400
    
    if user_name not in user_data:
        return jsonify({"error": f"No data found for user: {user_name}"}), 404

    # Retrieve user's stored embeddings and strings
    string_list = user_data[user_name]["string_list"]
    embeddings = user_data[user_name]["embeddings"]

    # Encode query string
    query_embedding = model.encode(query_string, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)
    
    # Find the most similar string
    most_similar_index = torch.argmax(cosine_scores).item()
    most_similar_string = string_list[most_similar_index]
    
    return jsonify({
        "query": query_string,
        "most_similar_string": most_similar_string,
        "cosine_score": cosine_scores[0][most_similar_index].item()
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
