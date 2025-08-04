import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class SimpleVocab:
    """Mapping token-index."""
    def __init__(self, counter, max_size, specials):
        self.itos = list(specials) + [
            w for w,_ in counter.most_common(max_size - len(specials))
        ]
        self.stoi = {w:i for i,w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<UNK>'])


class TorchTokenizer(BaseEstimator, TransformerMixin):
    """Fits a fold-local vocab and pads sequences to seq_len."""
    def __init__(self, num_words=30_000, seq_len=200):
        self.num_words = num_words
        self.seq_len   = seq_len

    def fit(self, X, y=None):
        counts = Counter(tok.lower() for doc in X for tok in doc.split())
        self.vocab = SimpleVocab(counts, self.num_words, specials=['<PAD>','<UNK>'])
        return self

    def transform(self, X):
        pad_id = self.vocab['<PAD>']
        seqs = [
            [self.vocab[tok.lower()] for tok in doc.split()[:self.seq_len]]
            for doc in X
        ]
        # pad to seq_len
        return np.array([
            s + [pad_id]*(self.seq_len - len(s)) for s in seqs
        ], dtype=np.int64)


class TextCNN(nn.Module):
    """A simple text CNN with adjustable output size."""
    def __init__(self,
                 vocab_size,
                 emb_matrix,
                 seq_len=200,
                 n_filters=128,
                 kernel_size=5,
                 dense_units=64,
                 dropout=0.5,
                 trainable=False,
                 n_classes=6):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(
            emb_matrix,
            freeze=not trainable,
            padding_idx=0
        )
        self.conv = nn.Conv1d(emb_matrix.size(1), n_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_filters, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, n_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len) → embed → (batch, seq_len, emb_dim)
        x = self.embed(x).transpose(1, 2)  # → (batch, emb_dim, seq_len)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x)
