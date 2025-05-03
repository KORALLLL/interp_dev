import argparse
import os

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EmbeddingsDataset(Dataset):
    def __init__(
            self, source_path, split, source_type, collection_name="intonation_embeddings"):
        if source_type == "chromadb":
            self.embeddings, self.labels = self.get_chroma_embeddings(
                source_path, split, collection_name)
        else:
            raise ValueError(
                f"Invalid source type: {source_type}. "
                "Choose 'chromadb'."
            )

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def get_chroma_embeddings(
            self, source_path, split, collection_name="intonation_embeddings"):
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={"split": split}, include=["embeddings", "metadatas"])
        embeddings = np.array(results['embeddings'], dtype=np.float32)
        labels = [[item['f1'], item['f2'], item['f3']] for item in results['metadatas']]

        return embeddings, labels

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)


class FormantPredictor(nn.Module):

    def __init__(self, input_dim=256, output_dim=3):
        super(FormantPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2


def train(model, train_loader, optimizer, criterion, num_epoch, device):
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        model.train()

        for embeddings_batch, labels_batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"
        ):
            embeddings_batch = embeddings_batch.to(device)

            _, outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for embeddings_batch, labels_batch in tqdm(
                test_loader, desc="Evaluation Progress"):
            embeddings_batch = embeddings_batch.to(device)

            _, outputs = model(embeddings_batch)

            true_labels.extend(labels_batch.numpy())
            pred_labels.extend(outputs.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    metrics = {
        "mse_f1": mean_squared_error(true_labels[:, 0], pred_labels[:, 0]),
        "mse_f2": mean_squared_error(true_labels[:, 1], pred_labels[:, 1]),
        "mse_f3": mean_squared_error(true_labels[:, 2], pred_labels[:, 2]),
        "r2_f1": r2_score(true_labels[:, 0], pred_labels[:, 0]),
        "r2_f2": r2_score(true_labels[:, 1], pred_labels[:, 1]),
        "r2_f3": r2_score(true_labels[:, 2], pred_labels[:, 2])
    }

    return metrics


def get_loaders(source_path, source_type):
    train_dataset = EmbeddingsDataset(
        source_path, split="train", source_type=source_type)
    test_dataset = EmbeddingsDataset(
        source_path, split="test", source_type=source_type)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (
        train_loader,
        test_loader,
        test_dataset,
        train_dataset.embeddings.shape[1]
    )


def save_visualization(model, vectors, labels, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        _, predicted = model(vectors)

    predicted = predicted.cpu().numpy()
    labels = np.array(labels)

    plt.figure(figsize=(15, 5))
    for i, formant in enumerate(['F1', 'F2', 'F3']):
        plt.subplot(1, 3, i+1)
        plt.scatter(labels[:, i], predicted[:, i], alpha=0.6)
        plt.plot([labels[:, i].min(), labels[:, i].max()], [labels[:, i].min(), labels[:, i].max()], 'r--')
        plt.xlabel(f'True {formant}')
        plt.ylabel(f'Predicted {formant}')
        plt.title(f'{formant} Comparison')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_source",
        type=str,
        choices=["chromadb"],
        required=True,
        help="Source for embeddings: chromadb"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="./intonation_contour/embeddings",
        help="Path to chromadb collection folder"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./intonation_contour/scores/formants.txt",
        help="Save path for evaluation results file (txt)"
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="./intonation_contour/result/formants.png",
        help="Save path for embeddings visualisation"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./intonation_contour/models/formant_predictor.pth",
        help="Path to save the trained model"
    )
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"Folder {args.source_path} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, test_dataset, input_dim = get_loaders(
        args.source_path, args.embeddings_source
    )
    model = FormantPredictor(input_dim, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train(model, train_loader, optimizer,
          criterion, num_epoch=300, device=device)

    metrics = evaluate(model, test_loader, device)
    save_metrics(metrics, args.eval_path)
    save_visualization(
        model, test_dataset.embeddings.numpy(),
        test_dataset.labels.numpy(), args.visual_path, device=device
    )
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_save_path)


if __name__ == '__main__':
    main()