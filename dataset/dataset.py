import logging

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding

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
        max_length: int = 512,
        use_huggingface: bool = True
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
        self.use_huggingface = use_huggingface

    def num_labels(self) -> int:
        nunique = self.dataframe["label"].nunique()
        if nunique.isinstance(int):
            return nunique
        else:
            return 0

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> tuple[BatchEncoding, int] | BatchEncoding:
        question: str = self.dataframe.iloc[index]["question"]
        label: int = self.dataframe.iloc[index]["label"]
        self.dataframe["label"].nunique()
        _LOGGER.info(f"Tokenizing: {question}")
        # print(
        #     colored(f"Tokenizing Question >>\n", "dark_grey"),
        #     colored(f"{question}", "blue"),
        #     colored(f"[{label}]", "red")
        # )

        tokenized_question: BatchEncoding = self.tokenizer.tokenize(
            question,
            max_length=self.max_length
        )
        
        if self.use_huggingface:
            tokenized_question["input_ids"] = tokenized_question["input_ids"].squeeze(0)
            torch.Tensor([label]).to(torch.int64)
            tokenized_question["labels"] = label # F.one_hot(index_tensor, num_classes=num_labels)
            return tokenized_question
        else:
            return (tokenized_question, label)
