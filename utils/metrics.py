import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import torch
from tqdm import tqdm


def evaluate_emb_model(model, test_loader, device):
    """
    Evaluates a model on a test dataset. Calculates accuracy,
    precision, recall and f1-score
    """
    model.eval()
    total_samples_test = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in tqdm(
                test_loader, desc="Evaluation Progress"):
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

    return metrics, true_labels, pred_labels


def evaluate_probing(layer, y_pred, y_true):
    """
    Computes classification metrics (accuracy and F1-score) for a given layer.

    Converts probabilistic y_pred predictions into binary labels, then
    compares them with the true y_true labels and computes the metrics.
    """
    y_pred_labels = (y_pred >= 0.5).astype(int).squeeze(1)
    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    return layer, {"accuracy": acc, "f1_score": f1}


def read_metrics(file_path):
    """
    Reads metrics from a text file saved in save_metrics format.
    """
    metrics_list = []
    current_layer = None
    current_metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_layer, current_metrics = None, {}
                continue
            if ':' not in line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_metrics = {}
                current_layer = line
            else:
                key, value = line.split(':', 1)
                current_metrics[key.strip()] = float(value.strip())
    if current_layer and current_metrics:
        metrics_list.append((current_layer, current_metrics))
    return metrics_list


def plot_metrics(metrics_list, save_path):
    """
    Builds Accuracy and F1-score graphs by layers
    and saves them to the specified file.

    The function builds two line graphs - one for accuracy
    and one for F1-score - by layers from metrics_list.
    X-axes are layer identifiers.
    """
    layers = [m[0] for m in metrics_list]
    accuracies = [m[1]["accuracy"] for m in metrics_list]
    f1_scores = [m[1]["f1_score"] for m in metrics_list]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(layers, accuracies, color='b', label="Accuracy")
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across layers")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(layers, f1_scores, color='g', label="F1-score")
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("Layers")
    plt.ylabel("F1-score")
    plt.title("F1-score across layers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def save_metrics(metrics_list, save_path):
    """
    Saves metrics by layer to a text file.

    For each layer, the metrics are written in the form:
        <layer>
        accuracy: <value>
        f1_score: <value>

    If the file already exists, new metrics are added to the end.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a') as f:
        for layer, metrics in metrics_list:
            f.write(f"{layer}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


def save_to_csv(chunk_rows, layer, save_path):
    """
    Saves or updates a CSV file with model predictions for a specified layer.

    The function takes a list of strings (predictions and labels)
    and saves them to a CSV file.
    If the file already exists:
        - Adds a column for the predictions of the corresponding layer
            if it does not exist.
        - Updates the true_label and prediction_<layer> values
            for existing records by file name.
        - Adds new rows if the filename does not exist in the current CSV.

    If the file does not exist, it will be created and
    populated with the passed rows.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_new = pd.DataFrame(chunk_rows)
    prediction_col = f"prediction_{layer}"

    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)

        if prediction_col not in df_existing.columns:
            df_existing[prediction_col] = np.nan

        for _, row in df_new.iterrows():
            filename = row["filename"]
            true_label = row["true_label"]
            prediction = row[prediction_col]

            if str(filename) in df_existing["filename"].values:
                idx = df_existing[df_existing["filename"] == filename].index[0]

                if pd.isna(df_existing.at[idx, "true_label"]):
                    df_existing.at[idx, "true_label"] = true_label

                if pd.isna(df_existing.at[idx, prediction_col]):
                    df_existing.at[idx, prediction_col] = prediction
            else:
                new_row = {col: np.nan for col in df_existing.columns}
                new_row["filename"] = filename
                new_row["true_label"] = true_label
                new_row[prediction_col] = prediction
                df_existing = pd.concat(
                    [df_existing, pd.DataFrame([new_row])], ignore_index=True)

        df_existing.to_csv(save_path, index=False)
    else:
        df_new.to_csv(save_path, index=False)
