from transformers import AutoTokenizer, BatchEncoding

from ..const import PRETRAIN_URL


class QuestionTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_URL)

    def tokenize(
        self,
        sequence: str,
        max_length: int
    ) -> BatchEncoding:
        tokenized_question: BatchEncoding = self.tokenizer(
            sequence,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return tokenized_question
