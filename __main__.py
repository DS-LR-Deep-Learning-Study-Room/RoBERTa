import logging

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from .dataset.dataset import QuestionDataset
from .const import (
    PRETRAIN_URL,
    TRAIN_SET,
    TEST_SET,
    VALID_SET
)

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_URL)

    train_dataset = QuestionDataset(filename=TEST_SET, tokenizer=tokenizer)
    valid_dataset = QuestionDataset(filename=VALID_SET, tokenizer=tokenizer)
    test_dataset = QuestionDataset(filename=TEST_SET, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset)
    valid_loader = DataLoader(valid_dataset)
    test_loader = DataLoader(test_dataset)
    a, b = next(iter(train_loader))

    print(a)
    print(b)
