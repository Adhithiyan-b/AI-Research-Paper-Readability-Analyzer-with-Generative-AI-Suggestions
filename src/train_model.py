import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocess import preprocess_text
from src.readability_features import extract_features_df, ReadabilityFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_dataset(filepath: str):
    """
    Generates a synthetic academic dataset for training if none exists.
    """
    logger.info("Generating synthetic dataset...")
    data = {
        "text": [
            # Easy
            "The dog sat on the mat. It was a good day. The sun was shining bright.",
            "Water is important for life. We need to drink water every day. Plants also need water.",
            "This book is very simple to read. It has short words and small sentences.",
            # Moderate
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "The industrial revolution brought significant changes to society, transitioning from hand production methods to machines and new chemical manufacturing processes.",
            "Regular physical activity is proven to help prevent and manage noncommunicable diseases such as heart disease, stroke, diabetes, and several cancers.",
            # Hard
            "Deep neural network architectures have significantly improved performance in complex pattern recognition tasks, leveraging backpropagation and non-linear activation functions to map high-dimensional latent spaces.",
            "The socio-economic implications of macroeconomic fiscal policy adjustments necessitate a paradigm shift in how quantitative easing is structured within inflationary environments.",
            "Epigenetic modifications, including DNA methylation and histone acetylation, play a crucial role in regulating gene expression without altering the underlying nucleotide sequence, thus contributing to cellular differentiation."
        ] * 5,  # Multiply to have more samples
        "label": ["Easy", "Easy", "Easy", "Moderate", "Moderate", "Moderate", "Hard", "Hard", "Hard"] * 5
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    logger.info(f"Synthetic dataset saved to {filepath}")
    return df

def train_model():
    dataset_path = "dataset/sample_texts.csv"
    model_path = "model/readability_model.pkl"
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found at {dataset_path}.")
        df = generate_synthetic_dataset(dataset_path)
    else:
        logger.info(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
    X = df['text'].tolist()
    y = df['label'].tolist()
    
    logger.info("Building the ML pipeline...")
    
    # Feature Union combines multiple feature extraction mechanisms
    features = FeatureUnion([
        ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, max_features=1000)),
        ('readability_stats', ReadabilityFeatureExtractor())
    ])
    
    # Complete Pipeline
    pipeline = Pipeline([
        ('features', features),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    logger.info("Training the model...")
    pipeline.fit(X, y)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()
