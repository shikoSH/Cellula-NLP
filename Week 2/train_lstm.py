"""
Train an LSTM classifier on the Cellula toxic-content dataset.

Fixes vs. the original notebook:
 - Actually trains on cellula_toxic_data.csv (the notebook loaded a
   different file, "train.csv", i.e. the Jigsaw dataset, and never
   touched the real project data).
 - Trains on the `query` column. (The `image descriptions` column only
   has 12 unique values total -- a small fixed pool of generic stock
   captions like "A child playing in a sunny meadow" reused across
   hundreds of rows under different labels. It carries no real signal
   about the label, so folding it into the training text -- as the
   notebook's own markdown plan proposed -- would just inject label
   noise and break stratified splitting on the rarer classes. At
   inference time this is fine: the LSTM just gets handed whatever
   text it's given, whether that's a raw user query or a BLIP caption.)
 - Multi-class (9 categories), not hardcoded binary.
 - Training loop is completed (the notebook's `for x_batch, len_batch,
   y_batch in val_loader:` loop was cut off mid-line and never computed
   val metrics or saved a checkpoint, so `best_lstm_model.pt` never
   actually got created).
 - Saves everything inference needs (weights, vocab, label map, max_len)
   into one artifact file so text_classifier.py doesn't depend on
   notebook globals.
 - Uses a trainable nn.Embedding instead of GloVe. GloVe download
   requires nlp.stanford.edu, which isn't reachable in this sandbox
   -- swap in the GloVe-init block (commented below) if you run this
   on your own machine/Colab with internet access.
"""
import re
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

CSV_PATH = "cellula_toxic_data.csv"
ARTIFACT_PATH = "lstm_artifacts.pt"
MAX_LEN = 40
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
BATCH_SIZE = 64
NUM_EPOCHS = 25
PATIENCE = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# 1. Load + stack query / image-description columns (same label)
# ---------------------------------------------------------------------
def load_and_stack(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    stacked = df[["query", "Toxic Category"]].rename(
        columns={"query": "text", "Toxic Category": "label"}
    )
    stacked = stacked.dropna(subset=["text", "label"])
    stacked["text"] = stacked["text"].astype(str)
    # NOTE: several rare classes (Suicide & Self-Harm, Elections,
    # Sex-Related Crimes, Child Sexual Exploitation) are almost entirely
    # ONE templated sentence copy-pasted 100+ times, with only 2-3 truly
    # unique queries each. Deduping on text (as the notebook's markdown
    # plan suggested) collapses those classes down to 2-3 rows total,
    # which is too few to stratify-split. We keep duplicates here and
    # rely on the train/val/test split below being stratified so the
    # same distribution shows up in every split.
    stacked = stacked.reset_index(drop=True)
    return stacked


# ---------------------------------------------------------------------
# 2. Classical NLP preprocessing: clean -> tokenize -> stopwords -> lemma
# ---------------------------------------------------------------------
def preprocess(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokenized_docs, keep_mask = [], []

    for t in texts:
        review = re.sub("[^a-zA-Z]", " ", str(t)).lower().split()
        review = [lemmatizer.lemmatize(w) for w in review if w not in stop_words]
        if len(review) > 0:
            tokenized_docs.append(review)
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    return tokenized_docs, keep_mask


def build_vocab(tokenized_docs):
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for doc in tokenized_docs:
        for word in doc:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    return word_to_idx


def encode_and_pad(tokenized_docs, word_to_idx, max_len):
    encoded, lengths = [], []
    for doc in tokenized_docs:
        ids = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in doc]
        lengths.append(min(len(ids), max_len))
        if len(ids) < max_len:
            ids = ids + [word_to_idx["<PAD>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        encoded.append(ids)
    return np.array(encoded), np.array(lengths)


# ---------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 pad_idx=0, dropout=0.4, embedding_matrix=None):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False, padding_idx=pad_idx
            )
        else:
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
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        logits = self.fc(self.dropout(hidden_cat))
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- data ----
    stacked_df = load_and_stack(CSV_PATH)
    print(f"Stacked corpus size: {len(stacked_df)}")

    tokenized_docs, keep_mask = preprocess(stacked_df["text"].values)
    stacked_df = stacked_df[keep_mask].reset_index(drop=True)
    print(f"After dropping empty docs: {len(stacked_df)}")

    labels_sorted = sorted(stacked_df["label"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_sorted)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    num_classes = len(label_to_idx)
    print(f"Classes ({num_classes}): {label_to_idx}")

    encoded_labels = stacked_df["label"].map(label_to_idx).values

    # ---- split BEFORE building vocab, to avoid test/val leakage ----
    idx_all = np.arange(len(tokenized_docs))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    train_docs = [tokenized_docs[i] for i in idx_train]
    word_to_idx = build_vocab(train_docs)
    print(f"Vocab size (train-only): {len(word_to_idx)}")

    all_docs = [tokenized_docs[i] for i in idx_all]
    X_all, len_all = encode_and_pad(all_docs, word_to_idx, MAX_LEN)

    X_train, len_train = X_all[idx_train], len_all[idx_train]
    X_val, len_val = X_all[idx_val], len_all[idx_val]
    X_test, len_test = X_all[idx_test], len_all[idx_test]

    # ---- class weights for imbalance ----
    class_counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    class_weights = {i: total / (num_classes * max(class_counts[i], 1)) for i in range(num_classes)}
    print("Class weights:", class_weights)

    def to_loader(X, lengths, y, shuffle, sampler=None):
        ds = TensorDataset(torch.LongTensor(X), torch.LongTensor(lengths), torch.LongTensor(y))
        if sampler is not None:
            return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    sample_weights = torch.DoubleTensor([class_weights[l] for l in y_train])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = to_loader(X_train, len_train, y_train, shuffle=False, sampler=sampler)
    val_loader = to_loader(X_val, len_val, y_val, shuffle=False)
    test_loader = to_loader(X_test, len_test, y_test, shuffle=False)

    # ---- model / loss / optim ----
    model = LSTMClassifier(
        vocab_size=len(word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        pad_idx=word_to_idx["<PAD>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for x_batch, len_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch, len_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ---- validation ----
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x_batch, len_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch, len_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1:02d} | train_loss={avg_train_loss:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best val_macro_f1={best_val_f1:.4f})")
                break

    # ---- restore best checkpoint, evaluate on test ----
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for x_batch, len_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch, len_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(y_batch.numpy())

    target_names = [idx_to_label[i] for i in range(num_classes)]
    print("\nTest set report:")
    print(classification_report(test_true, test_preds, target_names=target_names, zero_division=0))
    print("Confusion matrix:")
    print(pd.DataFrame(confusion_matrix(test_true, test_preds), index=target_names, columns=target_names))

    # ---- save everything inference needs ----
    torch.save(
        {
            "model_state_dict": best_state,
            "word_to_idx": word_to_idx,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "max_len": MAX_LEN,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": num_classes,
        },
        ARTIFACT_PATH,
    )
    print(f"\nSaved artifacts to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
