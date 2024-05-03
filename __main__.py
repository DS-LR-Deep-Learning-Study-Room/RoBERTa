from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from .dataset.dataset import QuestionDataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    train_dataset = QuestionDataset(
        filename="train.parquet",
        tokenizer=tokenizer
    )

    train_dataloader = DataLoader(dataset=train_dataset)
    a, b = next(iter(train_dataloader))

    print(a)
    print(b)
