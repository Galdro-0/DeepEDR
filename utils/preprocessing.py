import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import re

# Longueur max d'une séquence de tokens
MAX_SEQ_LENGTH = 30 
TOKENIZER_PATH = 'models/tokenizer.json'
DATA_PATH = 'data/generated_logs.csv'

def tokenize_sequence(sequence_str):
    """Sépare les commandes et les arguments basiquement."""
    # Remplacement simple pour séparer les chemins et les flags
    sequence_str = re.sub(r'([/\\]+)', ' \\1 ', sequence_str)
    sequence_str = re.sub(r'([=:-]+)', ' \\1 ', sequence_str)
    tokens = sequence_str.lower().split()
    return tokens

class Vocab:
    """Construit le vocabulaire (word -> index)"""
    def __init__(self, counter, max_size=1000):
        self.word_to_idx = {
            '<PAD>': 0,  # Padding
            '<UNK>': 1   # Mot inconnu
        }
        # Les mots les plus fréquents
        most_common = counter.most_common(max_size - 2)
        for i, (word, _) in enumerate(most_common, 2):
            self.word_to_idx[word] = i
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.word_to_idx, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            word_to_idx = json.load(f)
        
        vocab = cls(Counter()) # Dummy counter
        vocab.word_to_idx = word_to_idx
        vocab.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        vocab.vocab_size = len(word_to_idx)
        return vocab


class EDRDataset(Dataset):
    """Dataset PyTorch."""
    def __init__(self, sequences, labels, vocab, max_len):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence_str = self.sequences[idx]
        label = self.labels[idx]
        
        tokens = tokenize_sequence(sequence_str)
        
        # Conversion en indices
        indices = [self.vocab.word_to_idx.get(t, self.vocab.word_to_idx['<UNK>']) for t in tokens]
        
        # Padding / Truncation
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            padding_len = self.max_len - len(indices)
            indices = indices + [self.vocab.word_to_idx['<PAD>']] * padding_len
            
        return torch.tensor(indices), torch.tensor(label, dtype=torch.long)

def get_data_loaders(batch_size=32):
    """Fonction principale pour charger et préparer les DataLoaders."""
    df = pd.read_csv(DATA_PATH)
    
    # Construction du vocabulaire (sur le set d'entraînement)
    X_train, X_val, y_train, y_val = train_test_split(
        df['sequence'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )
    
    # 1. Construire le vocabulaire
    word_counter = Counter()
    for seq in X_train:
        word_counter.update(tokenize_sequence(seq))
    
    vocab = Vocab(word_counter, max_size=2000) # Vocabulaire de 2000 mots
    vocab.save(TOKENIZER_PATH)
    
    # 2. Créer les Datasets
    train_dataset = EDRDataset(X_train.tolist(), y_train.tolist(), vocab, MAX_SEQ_LENGTH)
    val_dataset = EDRDataset(X_val.tolist(), y_val.tolist(), vocab, MAX_SEQ_LENGTH)
    
    # 3. Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab