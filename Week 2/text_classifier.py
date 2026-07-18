import os
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_artifacts.pt")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_word_to_idx = None
_idx_to_label = None
_max_len = None
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


class LSTMClassifier(nn.Module):
    """Must match the architecture in train_lstm.py exactly, since we
    load its state_dict here."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx=0, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        logits = self.fc(self.dropout(hidden_cat))
        return logits


def _load_model():
    global _model, _word_to_idx, _idx_to_label, _max_len
    if _model is not None:
        return

    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Could not find {ARTIFACT_PATH}. Run `python train_lstm.py` first "
            "to train the model and produce this artifact."
        )

    checkpoint = torch.load(ARTIFACT_PATH, map_location=_device, weights_only=False)
    _word_to_idx = checkpoint["word_to_idx"]
    _idx_to_label = checkpoint["idx_to_label"]
    _max_len = checkpoint["max_len"]

    model = LSTMClassifier(
        vocab_size=len(_word_to_idx),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        pad_idx=_word_to_idx["<PAD>"],
    ).to(_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model


def _preprocess(text: str):
    review = re.sub("[^a-zA-Z]", " ", str(text)).lower().split()
    review = [_lemmatizer.lemmatize(w) for w in review if w not in _stop_words]
    return review


def classify_text(text: str) -> str:
    """Classify a piece of text (raw user query or an image caption)
    into one of the trained Toxic Category labels."""
    _load_model()

    tokens = _preprocess(text)
    if not tokens:
        return "Safe"  # empty/non-alphabetic input, nothing to flag

    ids = [_word_to_idx.get(w, _word_to_idx["<UNK>"]) for w in tokens]
    length = min(len(ids), _max_len)
    if len(ids) < _max_len:
        ids = ids + [_word_to_idx["<PAD>"]] * (_max_len - len(ids))
    else:
        ids = ids[:_max_len]

    x = torch.LongTensor([ids]).to(_device)
    lengths = torch.LongTensor([length])

    with torch.no_grad():
        logits = _model(x, lengths)
        pred_idx = torch.argmax(logits, dim=1).item()

    return _idx_to_label[pred_idx]
