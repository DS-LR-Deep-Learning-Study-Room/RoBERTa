from termcolor import colored
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer | None = None
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-10)

        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, epoch: int, num_classes: int) -> float:
        self.model.train()

        running_loss = 0.
        last_loss = 0.

        pbar = tqdm(self.data_loader, desc="Training", position=1)
        for index, inputs in enumerate(pbar):
            # input 형식 : [question, label]
            question, label = inputs[0].to(self.device), inputs[1].to(self.device)
            label = F.one_hot(label, num_classes=num_classes).float()

            self.optimizer.zero_grad()

            outputs = self.model(
                question["input_ids"].squeeze(1),
                attention_mask=question["attention_mask"]
            )

            loss = self.criterion(outputs.logits, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(running_loss = running_loss, loss = last_loss)
            if index % 100 == 99:
                last_loss = running_loss / 100
                running_loss = 0.

        pbar.close()
        return last_loss
