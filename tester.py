from termcolor import colored
from time import sleep

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Tester():
    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def test(self):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for index, inputs in enumerate(self.data_loader):
                # input 형식 : [question, label]
                question = inputs[0].to(self.device)
                label = inputs[1].to(self.device)

                # ?? : General output = (batch_size, max_length)
                # Currently (batch_size, 1, max_length) obtained.
                outputs = self.model(
                    question["input_ids"].squeeze(1),
                    attention_mask=question["attention_mask"]
                )

                props = torch.softmax(outputs.logits, dim=-1)
                predicted = props.argmax(-1)
                print(colored(f"predicted: {predicted.item()} label: {label.item()}", "cyan"))

                total_correct += (predicted == label).sum().item()

                if index % 100 == 0:
                    print(colored(f"Accuracy: {total_correct / len(self.data_loader)}", "light_green"))
        print(colored(f"Final Accuracy: {total_correct / len(self.data_loader)}", "green"))
