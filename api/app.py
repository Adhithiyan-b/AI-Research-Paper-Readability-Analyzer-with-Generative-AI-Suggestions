import logging
from flask import Flask, request, jsonify
from src.predict import predict_readability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint to analyze the readability of an academic paragraph.
    Expects JSON: {"text": "academic text..."}
    """
    logger.info("Received POST request to /analyze")
    
    # 1. Input validation
    if not request.is_json:
        logger.error("Request must be JSON")
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    if 'text' not in data:
        logger.error("Missing 'text' field in request")
        return jsonify({"error": "Missing 'text' field in JSON"}), 400
        
    text = data['text']
    
    if not isinstance(text, str) or not text.strip():
        logger.error("The 'text' field must be a non-empty string")
        return jsonify({"error": "The 'text' field must be a non-empty string"}), 400
        
    # 2. Prediction
    try:
        logger.info("Processing readability prediction...")
        result = predict_readability(text)
        logger.info(f"Prediction successful: {result['difficulty_level']}")
        
        # 3. Return structured JSON
        return jsonify(result), 200
        
    except ValueError as ve:
        logger.error(f"Prediction ValueError: {ve}")
        return jsonify({"error": str(ve)}), 503  # Service Unavailable (e.g., model not trained)
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == '__main__':
    # For local development demo support
    app.run(host='0.0.0.0', port=5000, debug=True)
