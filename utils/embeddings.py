import os
from pathlib import Path
from datasets import ClassificationEmbeddingsDataset

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch


def get_loaders(source_path, source_type):
    """
    Creates dataloaders for train and test files.
    """
    train_dataset = ClassificationEmbeddingsDataset(
        source_path, split="train", source_type=source_type)
    test_dataset = ClassificationEmbeddingsDataset(
        source_path, split="test", source_type=source_type)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (
        train_loader,
        test_loader,
        test_dataset,
        train_dataset.audio_data.shape[1]
    )


def assign_labels_by_parent_dir(embeddings):
    """
    Assigns labels to classes. In this case, by the name of the parent folder.
    """
    for emb in embeddings:
        class_name = Path(emb['file_path']).parent.name
        emb['label'] = class_name


def save_visualization(model, vectors, labels, save_path, device):
    """
    Saves embedding visualization in .png files.
    """
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
    return x1_reduced


def save_embs_to_csv(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame()
    df['filename'] = data['filenames']
    df['true_label'] = data['true_labels']
    df['x_coordinate'] = data['x_coord']
    df['y_coordinate'] = data['y_coord']

    df_preds = pd.DataFrame()
    df_preds['filename'] = data['filenames']
    df_preds['prediction'] = data['predictions']

    name_part, ext = save_path.rsplit('.', 1)
    preds_path = f"{name_part}_predicts.{ext}"

    df.to_csv(save_path, index=False)
    df_preds.to_csv(preds_path, index=False)


def save_emb_metrics(metrics, save_path):
    """
    Saves computed metrics in .txt file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
