import argparse

import torch
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from .const import MODEL_PATH, TEST_SET, TRAIN_SET, VALID_SET
from .dataset.dataset import QuestionDataset
from .dataset.tokenizer import QuestionTokenizer
from .network.roberta import fetch_RoBERTa_model, fetch_saved_RoBERTa_model
from .trainer import HuggingFaceTrainer, Trainer

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--eval", dest="eval", action="store_true", default=False)
parser.add_argument(
    "-E", "--epochs", dest="epochs", default=5, help="Number of epochs to train"
)
parser.add_argument(
    "-B", "--batch-size", dest="batch_size", default=8, help="Batch size of data loader"
)
parser.add_argument(
    "-F",
    "--use-huggingface",
    dest="huggingface",
    action="store_true",
    default=True,
    help="Use HuggingFace's Trainer class instead of pure PyTorch",
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = QuestionTokenizer()

    use_huggingface = args.huggingface
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)

    train_dataset = QuestionDataset(
        filename=TRAIN_SET, tokenizer=tokenizer, use_huggingface=use_huggingface
    )
    # Training 도중에 사용하는 평가 데이터 셋
    valid_dataset = QuestionDataset(
        filename=VALID_SET, tokenizer=tokenizer, use_huggingface=use_huggingface
    )
    # Fine-Tuning 후에 쓰이는 평가 데이터 셋
    test_dataset = QuestionDataset(
        filename=TEST_SET, tokenizer=tokenizer, use_huggingface=use_huggingface
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False)

    num_labels = train_dataset.num_labels()
    model = fetch_RoBERTa_model(num_labels=num_labels).to(device)

    if args.eval is False and args.huggingface is False:
        trainer = Trainer(model=model, data_loader=train_loader, device=device)

        current_epoch = 0
        pbar = tqdm(range(epochs), desc=f"Epoch {current_epoch}", position=0)
        for epoch in pbar:
            current_epoch = epoch
            trainer.train(epoch=epoch, num_classes=num_labels)

        torch.save(model, MODEL_PATH)
    elif args.eval is False and args.huggingface is True:
        trainer = HuggingFaceTrainer(
            model=model,
            train_data=train_dataset,
            eval_data=valid_dataset,
            epochs=float(epochs),
            batch_size=batch_size,
        )
        trainer.train()

    # Evaluate
    model = fetch_saved_RoBERTa_model(num_labels=num_labels)
    tester = HuggingFaceTrainer(model=model, batch_size=batch_size)
    result = tester.evaluate(eval_dataset=test_dataset)
    print(colored(result, "green"))
