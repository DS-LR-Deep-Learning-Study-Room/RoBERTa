import logging

import torch
from transformers import BatchEncoding
from torch.utils.data import DataLoader

from .dataset.dataset import QuestionDataset
from .dataset.tokenizer import QuestionTokenizer
from .const import (
    TRAIN_SET,
    TEST_SET,
    VALID_SET
)
from .network.roberta import fetch_RoBERTa_model
from .tester import Tester

_LOGGER = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    tokenizer = QuestionTokenizer()

    train_dataset = QuestionDataset(filename=TEST_SET, tokenizer=tokenizer)
    valid_dataset = QuestionDataset(filename=VALID_SET, tokenizer=tokenizer)
    test_dataset = QuestionDataset(filename=TEST_SET, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, shuffle=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)

    model = fetch_RoBERTa_model().to(device)

    tester = Tester(
        model=model,
        data_loader=test_loader,
        device=device
    )

    tester.test()

    # question: BatchEncoding
    # label: int
    # question, label = next(iter(train_loader))

    # print(question, label)

    # outputs = model(**question)
    # print(outputs.logit)
