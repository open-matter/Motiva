import torch

class RegressorModel(torch.nn.Module):
    def __init__(self, encoder, output_dim=21):
        super().__init__()
        self.encoder = encoder
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.encoder.config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids, attention_mask)
        pooled = self.pool(hidden.transpose(1, 2)).squeeze(-1)  # batch x hidden
        return self.regressor(pooled)
