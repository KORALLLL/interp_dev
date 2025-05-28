from .probing_model import ProbingCls

import torch
import torch.optim as optim
from tqdm import tqdm


def train_emb_model(
        model,
        train_loader,
        optimizer,
        criterion,
        num_epoch,
        device
):
    """
    Train a model on a train dataset
    """
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        model.train()

        for embeddings_batch, labels_batch, _ in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"
        ):
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.long()
            _, outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_probing_model(
        train_loader,
        input_dim,
        device,
        num_epoch=3,
        existing_model=None
):
    """
    Train a model on a train dataset.
    """
    model = existing_model or ProbingCls(input_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model
