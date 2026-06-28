import torch
from torch import nn


class IntentClassifier(nn.Module):
    """Character-level intent classifier + generic BIO slot tagger.

    Slots are reusable: FIELD_ID and FIELD_VALUE can appear N times.
    """

    def __init__(
        self,
        vocab_size: int,
        n_labels: int,
        n_intents: int,
        emb_dim: int = 96,
        hidden_dim: int = 160,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.slot_head = nn.Linear(hidden_dim * 2, n_labels)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_intents),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        mask = input_ids.ne(0).float()
        x = self.embedding(input_ids)
        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)
        slot_logits = self.slot_head(encoded)

        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        intent_logits = self.intent_head(pooled)
        return slot_logits, intent_logits
