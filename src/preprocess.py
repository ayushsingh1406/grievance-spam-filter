import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = ENGLISH_STOP_WORDS

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # remove phone numbers
    text = re.sub(r"\+?\d[\d\s-]{6,}\d", " ", text)

    # keep only alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess(text: str) -> str:
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


if __name__ == "__main__":
    s = "WIN a brand new phone! Click http://spam.example.com or call +91 9876543210"
    print(preprocess(s))
