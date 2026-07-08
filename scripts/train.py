import torch

from src.datasets import train_loader, val_loader
from src.models import ResNet18
from src.training import Trainer
from src.training.callbacks import Terminal

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = getattr(torch.nn, "CrossEntropyLoss")()

    trainer = Trainer(
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        callbacks=[Terminal()],
    )

    trainer.fit(model)
