import torch
from transformers import RobertaModel

class EncoderRoberta(torch.nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
