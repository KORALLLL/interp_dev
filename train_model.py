import argparse
import os

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class GenderCls(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(GenderCls, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2


def train(model, train_loader, optimizer, criterion, num_epoch, device):
    for epoch in tqdm(range(num_epoch)):
        model.train()

        for embeddings_batch, labels_batch in train_loader:
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.long()
            _, outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    total_samples_test = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for embeddings_batch, labels_batch in test_loader:
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.long()
            x1, outputs = model(embeddings_batch)

            total_samples_test += 1

            _, predicted = torch.max(outputs.cpu(), 1)
            true_labels.extend(labels_batch.numpy())
            pred_labels.extend(predicted.numpy())

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1_score": f1_score(true_labels, pred_labels)
    }

    return metrics


def get_npy_embeddings(source_path):
    source = np.load(os.path.join(
        source_path, "numpy_embs.npy"), allow_pickle=True)
    source = source[0]

    train_embeddings = np.array([item['embedding']
                                for item in source['train']])
    train_labels = [item['label'] for item in source['train']]

    test_embeddings = np.array([item['embedding'] for item in source['test']])
    test_labels = [item['label'] for item in source['test']]
    return train_embeddings, train_labels, test_embeddings, test_labels


def get_chroma_embeddings(source_path, split, collection_name="gender_embeddings"):
    client = chromadb.PersistentClient(path=source_path)
    collection = client.get_collection(name=collection_name)
    results = collection.get(where={"split": split}, include=[
                             "embeddings", "metadatas"])
    embeddings = np.array(results['embeddings'], dtype=np.float32)
    labels = [item['label'] for item in results['metadatas']]

    return embeddings, labels


def get_loaders(train_vectors, train_labels, test_vectors, test_labels):

    train_dataset = TensorDataset(torch.tensor(
        train_vectors), torch.tensor(train_labels))
    test_dataset = TensorDataset(torch.tensor(
        test_vectors), torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def save_visualization(model, vectors, labels, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vectors = torch.FloatTensor(vectors).to(device)
    with torch.no_grad():
        x1, predicted = model(vectors)

    reducer = TSNE(n_components=2, random_state=42)
    x1_reduced = reducer.fit_transform(x1.detach().cpu().numpy())

    unique_labels = list(set(labels))

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            x1_reduced[indices, 0],
            x1_reduced[indices, 1],
            label=f"Label: {label}",
            alpha=0.6
        )

    plt.title("Visualization of embeddings after first layer")
    plt.legend()
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
        choices=["npy", "chromadb"],
        required=True,
        help="Source for embeddings: npy or chromadb"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="./embeddings",
        help="Path to npy file or to chromadb collection folder"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./scores/gender.txt",
        help="Save path for evaluation results file (txt)"
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="./result/gender.png",
        help="Save path for embeddings visualisation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"Folder {args.source_path} does not exists.")

    if args.embeddings_source == "npy":
        train_embeddings, train_labels, test_embeddings, test_labels = get_npy_embeddings(
            args.source_path)
    else:
        train_embeddings, train_labels = get_chroma_embeddings(
            args.source_path, split="train")
        test_embeddings, test_labels = get_chroma_embeddings(
            args.source_path, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lb = LabelEncoder()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    train_loader, test_loader = get_loaders(
        train_embeddings, train_labels, test_embeddings, test_labels)
    model = GenderCls(train_embeddings.shape[1], 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(model, train_loader, optimizer,
          criterion, num_epoch=300, device=device)

    metrics = evaluate(model, test_loader, device)
    save_metrics(metrics, args.eval_path)
    save_visualization(model, test_embeddings, test_labels,
                       args.visual_path, device=device)


if __name__ == '__main__':
    main()
