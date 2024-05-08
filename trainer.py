from termcolor import colored

import torch
import torch.nn as nn
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

        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch: int) -> float:
        self.model.train()

        running_loss = 0.
        last_loss = 0.

        for index, inputs in enumerate(self.data_loader):
            # input 형식 : [question, label]
            question = inputs[0].to(self.device)
            label = inputs[1].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                question["input_ids"].squeeze(1),
                attention_mask=question["attention_mask"]
            )
            print(outputs.logits)

            loss = self.criterion(outputs.logits, label.float())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if index % 100 == 99:
                last_loss = running_loss / 100
                print(colored(f"epoch {epoch}) batch {index + 1} loss: {last_loss}", "dark_grey"))
                running_loss = 0.

        return last_loss
