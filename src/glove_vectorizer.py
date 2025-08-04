import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GloveVectorizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that averages GloVe word vectors
    to produce a fixed-length document embedding.
    """

    def __init__(self, glove_path=None, lowercase=True, tokenizer=None):
        """
        Parameters:
        glove_path : str
            Path to the GloVe .txt file (one “word value value …” line per token).
        lowercase : bool, default=True
            Whether to lowercase incoming text.
        tokenizer : callable, default=None
            Function mapping a string → list of tokens; if None, splits on whitespace.
        """
        self.glove_path = glove_path
        self.lowercase = lowercase
        self.tokenizer = tokenizer or (lambda text: text.split())
        self.embedding_dim = None
        self.embeddings_ = {}

    def fit(self, X, y=None):
        # Load the embeddings file into memory
        if not os.path.exists(self.glove_path):
            raise FileNotFoundError(f"GloVe file not found: {self.glove_path}")
        with open(self.glove_path, 'r', encoding='utf8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                if self.embedding_dim is None:
                    self.embedding_dim = vec.shape[0]
                self.embeddings_[word] = vec
        return self

    def transform(self, X):
        # For each document, average all token vectors (or zero-vector if none found)
        docs = []
        for doc in X:
            text = doc.lower() if self.lowercase else doc
            tokens = self.tokenizer(text)
            vecs = [self.embeddings_[tok] for tok in tokens if tok in self.embeddings_]
            if vecs:
                doc_vec = np.mean(vecs, axis=0)
            else:
                doc_vec = np.zeros(self.embedding_dim, dtype=np.float32)
            docs.append(doc_vec)
        return np.vstack(docs)