import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class TextClassificationModel(nn.Module):
    def __init__(self, num_class):
        super(TextClassificationModel, self).__init__()

        # Load BERT model and tokenizer
        self.bert = BertModel.from_pretrained("prajjwal1/bert-mini")
        self.bert_tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

        # Classification layer
        self.fc = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, text):
        # Obtain BERT embeddings
        ex_ids = self.bert_tokenizer.encode(text, add_special_tokens=True)
        reps = self.bert(torch.tensor([ex_ids]))
        pooled = reps.pooler_output

        # Perform classification
        return self.fc(pooled)
