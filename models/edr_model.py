import torch
import torch.nn as nn

class LSTMEDR(nn.Module):
    """
    Modèle LSTM pour la classification de séquences de commandes.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size (int): Taille du vocabulaire (obtenu du tokenizer).
            embedding_dim (int): Dimension des embeddings (ex: 64, 128).
            hidden_dim (int): Dimension des couches cachées du LSTM (ex: 128).
            num_classes (int): Nombre de classes de sortie (ex: 4).
            num_layers (int): Nombre de couches LSTM superposées.
            dropout (float): Taux de dropout.
        """
        super(LSTMEDR, self).__init__()
        
        self.num_classes = num_classes
        
        # 1. Couche d'Embedding
        # Transforme les indices (mots) en vecteurs denses
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Couche LSTM
        # batch_first=True signifie que l'input est (batch_size, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True # Bidirectionnel pour capturer le contexte avant/après
        )
        
        # 3. Couche de Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. Couche de Classification (Feed-Forward)
        # hidden_dim * 2 car le LSTM est bidirectionnel
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """Passe avant (forward pass)."""
        
        # x shape: (batch_size, seq_len)
        
        # 1. Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # 2. LSTM
        # output contient les états cachés de chaque pas de temps
        # (h_n, c_n) sont les derniers états cachés et cellulaires
        lstm_output, (hidden_state, cell_state) = self.lstm(embedded)
        
        # Nous voulons l'état caché final du LSTM bidirectionnel.
        # On concatène le dernier état "forward" (hidden_state[-2]) et "backward" (hidden_state[-1])
        # hidden_state shape: (num_layers * num_directions, batch_size, hidden_dim)
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim=1)
        
        # hidden shape: (batch_size, hidden_dim * 2)
        
        # 3. Dropout
        hidden_dropped = self.dropout(hidden)
        
        # 4. Classification
        logits = self.fc(hidden_dropped)
        # logits shape: (batch_size, num_classes)
        
        return logits