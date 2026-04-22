import pytest
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_analyze_endpoint_success(client, monkeypatch):
    """
    Test the /analyze endpoint with valid input.
    """
    # Mocking the pipeline is better for unit tests, but since we train a small model,
    # we can just let it run if the model exists. For robust tests, we'd mock it.
    
    # We will mock predict_readability to avoid needing the model file in simple API tests.
    from api import app as api_app
    
    def mock_predict(text):
        return {
            "readability_score": 42.5,
            "difficulty_level": "Hard",
            "confidence": 0.87,
            "ai_suggestion": "The paragraph contains long sentences and complex terminology. Consider shortening sentences and explaining technical terms."
        }
        
    monkeypatch.setattr("api.app.predict_readability", mock_predict)

    response = client.post('/analyze', json={
        "text": "Deep neural network architectures have significantly improved performance in complex pattern recognition tasks."
    })
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert "readability_score" in data
    assert "difficulty_level" in data
    assert "confidence" in data
    assert "ai_suggestion" in data
    assert data["difficulty_level"] in ["Easy", "Moderate", "Hard"]

def test_analyze_endpoint_missing_text(client):
    """
    Test the /analyze endpoint with missing text field.
    """
    response = client.post('/analyze', json={"wrong_field": "some text"})
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_analyze_endpoint_not_json(client):
    """
    Test the /analyze endpoint with non-JSON input.
    """
    response = client.post('/analyze', data="This is just plain text")
    assert response.status_code == 400
    assert "error" in response.get_json()
