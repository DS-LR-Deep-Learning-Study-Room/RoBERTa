import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from ..const import PRETRAIN_URL

def fetch_RoBERTa_model(num_labels: int) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        PRETRAIN_URL,
        num_labels=num_labels
    )
