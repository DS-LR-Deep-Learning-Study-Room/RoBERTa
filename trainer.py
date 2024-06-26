import contextlib
import os
from typing import Optional

import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers.trainer as hf_trainer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrainingArguments

from .const import LOGGING_PATH, MODEL_PATH, TRAINER_PATH


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        optimizer: Optional[optim.Optimizer] = None
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
            question = inputs["inputs"].to(self.device)
            label = inputs["labels"].to(self.device)
            label = F.one_hot(label, num_classes=num_classes).float()

            self.optimizer.zero_grad()

            outputs = self.model(
                question["input_ids"],
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

class HuggingFaceTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: Optional[Dataset] = None,
        eval_data: Optional[Dataset] = None,
        epochs: float = 3,
        batch_size: int = 8,
        label_names: Optional[list[str]] = None
    ):
        training_args = TrainingArguments(
            output_dir=TRAINER_PATH,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            label_names=label_names,
            load_best_model_at_end=True,
            logging_dir=LOGGING_PATH
        )
        self.trainer: hf_trainer.Trainer = hf_trainer.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=self.compute_metrics
        )
        
        self.metric = evaluate.load("accuracy")
    
    def compute_metrics(self, eval_pred) -> Optional[dict]:
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def train(self):
        self.trainer.train()
        
        with contextlib.suppress(OSError):
            os.remove(MODEL_PATH)
        self.trainer.save_model(MODEL_PATH)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> dict[str, float]:
        return self.trainer.evaluate(eval_dataset=eval_dataset)
