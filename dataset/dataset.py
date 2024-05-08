import logging
from termcolor import colored

from transformers import BatchEncoding

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import QuestionTokenizer

_LOGGER = logging.getLogger(__name__)

class QuestionDataset(Dataset):
    """
    Questions Parquet 파일로부터 Dataset을 생성합니다.
    `def __init__(self, filename: str)`
    """
    def __init__(
        self,
        filename: str,
        tokenizer: QuestionTokenizer,
        max_length: int = 512
    ):
        """
        `filename` : Parquet 파일 이름
        `tokenizer` : Tokenizer 인스턴스
        `max_length` : Token 최대 길이
        """
        super(QuestionDataset, self).__init__()

        self.dataframe = pd.read_parquet(
            "RoBERTa/dataset/resources/" + filename
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> tuple[BatchEncoding, int]:
        question: str = self.dataframe.iloc[index]["question"]
        label: int = self.dataframe.iloc[index]["label"]
        _LOGGER.info(f"Tokenizing: {question}")
        # print(
        #     colored(f"Tokenizing Question >>\n", "dark_grey"),
        #     colored(f"{question}", "blue"),
        #     colored(f"[{label}]", "red")
        # )

        # (batch_size, 1, max_length)
        tokenized_question: BatchEncoding = self.tokenizer.tokenize(
            question,
            max_length=self.max_length
        )

        return tokenized_question, label