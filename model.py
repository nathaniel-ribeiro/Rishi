import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(SinusoidalPositionalEncoding, self).__init__()
        positions = torch.arange(max_seq_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=256, n_layers=8, n_heads=8, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # re-initialize since TransformerEncoder initializes all layers with the same weights
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=torch.sqrt(2.0 / (module.embedding_dim + self.vocab_size)))
    
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x