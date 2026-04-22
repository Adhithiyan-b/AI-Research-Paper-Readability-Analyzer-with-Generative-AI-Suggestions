import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Preprocesses academic text by applying:
    - lowercase conversion
    - punctuation removal
    - whitespace normalization
    - stopword removal
    - tokenization and re-joining
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase conversion
    text = text.lower()

    # 2. Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Re-join for sklearn TF-IDF vectorizer which expects string, not list of tokens
    return ' '.join(filtered_tokens)

def preprocess_for_readability(text: str) -> str:
    """
    A lighter preprocessing for readability feature extraction.
    We don't want to remove punctuation or stopwords here because
    readability formulas depend on sentence boundaries and word counts.
    """
    if not isinstance(text, str):
        return ""
    # Just normalize whitespace
    return re.sub(r'\s+', ' ', text).strip()
