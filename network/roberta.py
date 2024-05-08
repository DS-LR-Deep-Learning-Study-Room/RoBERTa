import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from ..const import PRETRAIN_URL

def fetch_RoBERTa_model() -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(PRETRAIN_URL)
