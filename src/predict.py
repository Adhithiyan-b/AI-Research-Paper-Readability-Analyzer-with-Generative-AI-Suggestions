from src.readability_features import ReadabilityFeatureExtractor
import os
import joblib
import logging
import textstat
import google.generativeai as genai

# Required for unpickling the custom transformer
from src.preprocess import preprocess_for_readability

logger = logging.getLogger(__name__)

class ReadabilityPredictor:
    def __init__(self, model_path="model/readability_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self._load_model()
        self._setup_genai()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            logger.error(f"Model file not found at {self.model_path}")
            self.pipeline = None

    def _setup_genai(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        self.use_gemini = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use a compatible generative model
                self.gen_model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_gemini = True
                logger.info("Gemini API initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}. Falling back to rule-based suggestions.")
        else:
            logger.info("No GEMINI_API_KEY found. Using rule-based suggestions.")

    def get_readability_score(self, text: str) -> float:
        """
        Calculates a numeric readability score between 0 and 100.
        Uses Flesch Reading Ease bounded between 0-100.
        Higher score = Easier to read.
        """
        clean_text = preprocess_for_readability(text)
        flesch = textstat.flesch_reading_ease(clean_text)
        # Bound the score between 0 and 100
        score = max(0.0, min(100.0, flesch))
        return round(score, 2)

    def generate_suggestion(self, text: str, difficulty: str) -> str:
        if difficulty == "Easy":
            return "The text is already easy to read. No major changes needed."

        if self.use_gemini:
            try:
                prompt = (
                    "Analyze the following academic paragraph and suggest ways "
                    "to simplify its readability for students. Keep it concise.\n\n"
                    f"Text:\n{text}"
                )
                response = self.gen_model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Fall through to rule-based

        # Rule-based fallback
        if difficulty == "Hard":
            return "The paragraph contains long sentences and complex terminology. Consider shortening sentences and explaining technical terms."
        else:
            return "Consider breaking down some of the longer sentences and replacing advanced vocabulary with simpler synonyms to improve readability."

    def predict_readability(self, text: str) -> dict:
        if not self.pipeline:
            raise ValueError("Model pipeline is not loaded. Please train the model first.")
        
        # Predict difficulty and confidence
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = round(float(max(probabilities)), 2)
        
        # Get readability score
        score = self.get_readability_score(text)
        
        # Get suggestion
        suggestion = self.generate_suggestion(text, prediction)
        
        return {
            "readability_score": score,
            "difficulty_level": str(prediction),
            "confidence": confidence,
            "ai_suggestion": suggestion
        }

# Singleton instance
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ReadabilityPredictor()
    return _predictor_instance

def predict_readability(text: str) -> dict:
    predictor = get_predictor()
    return predictor.predict_readability(text)
