from termcolor import colored

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
                print(inputs)
                # input 형식 : [question, label]
                question = inputs["input_ids"].to(self.device)
                label = inputs["labels"].to(self.device)

                outputs = self.model(
                    question["input_ids"],
                    attention_mask=question["attention_mask"]
                )

                props = torch.softmax(outputs.logits, dim=-1)
                predicted = props.argmax(-1)
                print(colored(f"predicted: {predicted.item()} label: {label.item()}", "cyan"))

                total_correct += (predicted == label).sum().item()

                if index % 100 == 0:
                    print(colored(f"Accuracy: {total_correct / len(self.data_loader)}", "light_green"))
        print(colored(f"Final Accuracy: {total_correct / len(self.data_loader) * 100}", "green"))
