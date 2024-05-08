import argparse
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .dataset.dataset import QuestionDataset
from .dataset.tokenizer import QuestionTokenizer
from .const import (
    MODEL_PATH,
    TRAIN_SET,
    TEST_SET,
    VALID_SET
)
from .network.roberta import fetch_RoBERTa_model
from .tester import Tester
from .trainer import Trainer

_LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--test-only", dest="test", action="store_true", default=False)
parser.add_argument("-E", "--epochs", dest="epochs", default=5, help="Number of epochs to train")
parser.add_argument("-B", "--batch-size", dest="batch_size", default=8, help="Batch size of data loader")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = QuestionTokenizer()

    train_dataset = QuestionDataset(filename=TRAIN_SET, tokenizer=tokenizer)
    valid_dataset = QuestionDataset(filename=VALID_SET, tokenizer=tokenizer)
    test_dataset = QuestionDataset(filename=TEST_SET, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)

    num_labels = train_dataset.num_labels()
    model = fetch_RoBERTa_model(
        num_labels=num_labels
    ).to(device)

    if args.test == False:
        trainer = Trainer(
            model=model,
            data_loader=train_loader,
            device=device
        )
        
        current_epoch = 0
        pbar = tqdm(range(int(args.epochs)), desc=f"Epoch {current_epoch}", position=0)
        for epoch in pbar:
            current_epoch = epoch
            trainer.train(epoch=epoch, num_classes=num_labels)

        torch.save(model, MODEL_PATH)

    model = torch.load(MODEL_PATH).to(device)
    tester = Tester(
        model=model,
        data_loader=test_loader,
        device=device
    )
    tester.test()
