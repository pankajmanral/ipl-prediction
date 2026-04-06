from flask import Flask, send_from_directory, jsonify
import os
import json

app = Flask(__name__, static_folder='.')

@app.route('/')
def home():
    """Serves the main IPL prediction dashboard."""
    return send_from_directory('.', 'index.html')

@app.route('/predictions')
def predictions():
    """Allows clients to fetch the live prediction data."""
    try:
        with open('predictions_2026.json', 'r') as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"error": "Predictions not available on server yet"}), 404

@app.route('/<path:path>')
def static_proxy(path):
    """Serves all static assets (JSON, CSS, Images, etc.) from the root."""
    return send_from_directory('.', path)

if __name__ == '__main__':
    # Local dev server
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
