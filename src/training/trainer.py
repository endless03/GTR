import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from src.training.detection import adjust_threshold, calculate_sim_median_std
from src.utils.evaluation import evaluating_change_point


class Trainer:
    """Trainer class"""

    def __init__(self, model, train_loader, val_loader,
                 learning_rate=4.5e-4, weight_decay=1e-4,
                 epochs=100, early_stop_patience=40,
                 checkpoint_dir="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience

        # Optimizer
        self.optimizer = AdamW(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        self.loss_fn = nn.HuberLoss()

        # Checkpoint
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Track best model
        self.best_val_f1 = -float('inf')
        self.early_stop_counter = 0

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        for x, y in self.train_loader:
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()

            # Gradient amplification for relation encoder
            for name, param in self.model.named_parameters():
                if "relation_encoder" in name and param.grad is not None:
                    param.grad *= 2.0

            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def validate(self, val_anomalys):
        """Validate model"""
        threshold, val_f1, mean, std = adjust_threshold(
            self.model, self.val_loader, val_anomalys)
        return threshold, val_f1, mean, std

    def train(self, val_anomalys):
        """Complete training pipeline"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(self.checkpoint_dir,
                                       f"best_model_{timestamp}.pth")

        for epoch in range(self.epochs):
            # Training
            train_loss = self.train_epoch()

            # Validation
            threshold, val_f1, mean, std = self.validate(val_anomalys)

            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"Threshold: {threshold:.4f} | Val F1: {val_f1:.4f}")

            # Save best model
            if val_f1 > self.best_val_f1:
                print(f"New best F1 ({self.best_val_f1:.4f} -> {val_f1:.4f}), "
                      f"saving model... Threshold: {threshold:.4f}")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'threshold': threshold,
                    'val_f1': val_f1
                }, best_model_path)

                self.best_val_f1 = val_f1
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

        # Load best model
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.threshold = checkpoint['threshold']
        print(f"Loaded best model with threshold: {self.model.threshold:.4f}")

        return self.model