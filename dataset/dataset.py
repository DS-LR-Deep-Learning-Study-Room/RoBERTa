import logging

from termcolor import colored

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_LOGGER = logging.getLogger(__name__)

class QuestionDataset(Dataset):
    """
    Questions Parquet 파일로부터 Dataset을 생성합니다.
    `def __init__(self, filename: str)`
    """
    def __init__(self, filename: str, tokenizer):
        """
        `filename` : Parquet 파일 이름
        """
        super(QuestionDataset, self).__init__()

        self.dataframe = pd.read_parquet(
            "Encoder/dataset/resources/" + filename
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        question = self.dataframe.iloc[index]["question"]
        label = self.dataframe.iloc[index]["label"]
        _LOGGER.info(f"Tokenizing: {question}")
        print(
            colored(f"Tokenizing Question >>\n", "dark_grey"),
            colored(f"{question}", "blue"),
            colored(f"[{label}]", "red")
        )

        question = self.tokenizer.encode(question, return_tensors="pt")
        label = torch.tensor(label, dtype=torch.float32)

        return question, label
