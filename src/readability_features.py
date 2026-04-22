import textstat
import numpy as np
import pandas as pd
from src.preprocess import preprocess_for_readability
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

class ReadabilityFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            clean_text = preprocess_for_readability(text)
            if not clean_text:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            flesch = textstat.flesch_reading_ease(clean_text)
            gunning_fog = textstat.gunning_fog(clean_text)
            smog = textstat.smog_index(clean_text)
            
            word_count = textstat.lexicon_count(clean_text, removepunct=True)
            sentence_count = textstat.sentence_count(clean_text)
            avg_word_count = word_count / sentence_count if sentence_count > 0 else word_count

            features.append([flesch, gunning_fog, smog, avg_word_count])
            
        return np.array(features)

def extract_features(text: str) -> dict:
    """
    Computes linguistic complexity features for a given text.
    Returns a dictionary of features.
    """
    # Use minimally preprocessed text to preserve sentences and punctuation
    # for accurate readability metric calculations.
    clean_text = preprocess_for_readability(text)
    
    if not clean_text:
        return {
            'flesch_reading_ease': 0.0,
            'gunning_fog': 0.0,
            'smog_index': 0.0,
            'syllable_count': 0,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0,
            'lexical_diversity': 0.0
        }

    # Textstat metrics
    flesch = textstat.flesch_reading_ease(clean_text)
    gunning_fog = textstat.gunning_fog(clean_text)
    smog = textstat.smog_index(clean_text)
    syllables = textstat.syllable_count(clean_text)
    
    word_count = textstat.lexicon_count(clean_text, removepunct=True)
    sentence_count = textstat.sentence_count(clean_text)
    
    # Avoid division by zero
    word_count = word_count if word_count > 0 else 1
    sentence_count = sentence_count if sentence_count > 0 else 1
    
    # Custom metrics
    avg_sentence_length = word_count / sentence_count
    
    tokens = word_tokenize(clean_text.lower())
    words_only = [t for t in tokens if t.isalpha()]
    
    if words_only:
        avg_word_length = sum(len(w) for w in words_only) / len(words_only)
        unique_words = len(set(words_only))
        lexical_diversity = unique_words / len(words_only)
    else:
        avg_word_length = 0.0
        lexical_diversity = 0.0

    return {
        'flesch_reading_ease': flesch,
        'gunning_fog': gunning_fog,
        'smog_index': smog,
        'syllable_count': syllables,
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'lexical_diversity': lexical_diversity
    }

def extract_features_df(texts: list) -> pd.DataFrame:
    """
    Extracts features for a list of texts and returns a Pandas DataFrame.
    Useful for integration with Scikit-learn pipelines.
    """
    features_list = [extract_features(t) for t in texts]
    return pd.DataFrame(features_list)
