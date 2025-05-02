import argparse
import hashlib
import os
from pathlib import Path
from extract_features import extract_features

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wespeaker


class MaxActivationLength():
    def __init__(self):
        self.max_length = 0

    def update(self, layer_name, activation):
        if "pooling" in layer_name:
            return
        if activation.dim() == 4:
            current_length = activation.size(3)
        elif activation.dim() == 3:
            current_length = activation.size(2)
        else:
            return

        if current_length > self.max_length:
            self.max_length = current_length


class ActivationDataset(Dataset):
    def __init__(self, activation_paths, labels, max_len):
        self.activation_paths = activation_paths
        self.y = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len

    def __getitem__(self, idx):
        activation = np.load(self.activation_paths[idx])
        return self.pad_activation(activation), self.y[idx]

    def __len__(self):
        return len(self.y)

    def pad_activation(self, activation):
        tensor = torch.from_numpy(activation)
        if tensor.dim() == 2:
            if tensor.size(1) < self.max_len:
                pad = (0, self.max_len - tensor.size(1))
                tensor = torch.nn.functional.pad(tensor, pad)
            else:
                tensor = tensor[:, :self.max_len]
        return tensor.mean(dim=1)


class GetActivations(nn.Module):
    """
    Class for getting activations from a model.
    """

    def __init__(self, model):
        super(GetActivations, self).__init__()
        self.model = model

    def forward(self, x):
        out = x.permute(0, 2, 1)
        activations = []
        model_front = self.model.model.front

        x = out.unsqueeze(dim=1)

        out = model_front.relu(model_front.bn1(model_front.conv1(x)))
        activations.append({"first relu": out})

        for name, layer in model_front.named_children():
            c_sim = 0
            c_relu = 0
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                for sec_name, sec_layer in layer.named_children():
                    identity = out

                    out = sec_layer.relu(sec_layer.bn1(sec_layer.conv1(out)))
                    c_relu += 1
                    activations.append({f"{name} relu {c_relu}": out})

                    out = sec_layer.bn2(sec_layer.conv2(out))
                    out = sec_layer.SimAM(out)
                    c_sim += 1
                    activations.append({f"{name} SimAM {c_sim}": out})

                    if sec_layer.downsample is not None:
                        identity = sec_layer.downsample(identity)

                    out += identity
                    out = sec_layer.relu(out)
                    c_relu += 1
                    activations.append({f"{name} relu {c_relu}": out})

        out = self.model.model.pooling(out)
        activations.append({"pooling": out})

        if self.model.model.drop:
            out = self.model.model.drop(out)

        out = self.model.model.bottleneck(out)

        return activations, out


class EmotionCls(nn.Module):
    def __init__(self, input_dim=256, num_classes=5):
        super(EmotionCls, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.fc2(x1)
        return x2


def get_audio_path(audio_dir):
    """
    Recursively finds all audio files in the specified directory.
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('**/*.wav'))
    return audio_files


def get_activations(model, audio_path, device):
    """
    Gets model activations.
    """
    feats = extract_features(audio_path)
    feats = feats.to(device)

    with torch.no_grad():
        activations, _ = model(feats)

    acts = {
        "file_path": audio_path,
        "act": activations
    }
    return acts


def get_max_length(model, audio_files, device):
    max_activation_length = MaxActivationLength()

    for audio_path in audio_files:
        feats = extract_features(audio_path).to(device)
        with torch.no_grad():
            activations, _ = model(feats)

        for act_dict in activations:
            for layer_name, activation in act_dict.items():
                max_activation_length.update(layer_name, activation)
    return max_activation_length.max_length


def get_activations_for_layer(model, audio_files, device, layer_name, max_act_len):
    """
    Gets model activations for a specified layer.
    """
    label_encoder = LabelEncoder()
    labels = [Path(f).parent.parent.name for f in audio_files]
    labels = label_encoder.fit_transform(labels)

    save_dir = Path("activations") / layer_name
    save_dir.mkdir(parents=True, exist_ok=True)

    activation_paths = []
    filtered_labels = []

    with torch.no_grad():
        for k, audio_path in enumerate(tqdm(
            audio_files,
            desc=f"Extracting {layer_name} activations"
        )):
            feats = extract_features(audio_path).to(device)
            acts, _ = model(feats)

            activation = next((d[layer_name]
                              for d in acts if layer_name in d), None)
            if activation is None:
                continue

            if "pooling" in layer_name:
                activation = activation.flatten()
            else:
                activation = activation.squeeze(0)
                if activation.dim() == 3:
                    activation = activation.mean(dim=1)
                    if activation.size(1) < max_act_len:
                        pad = (0, max_act_len - activation.size(1))
                        activation = torch.nn.functional.pad(activation, pad)
                    else:
                        activation = activation[:, :max_act_len]

            str_audio_path = str(Path(audio_path).resolve())
            hash_object = hashlib.sha256()
            hash_object.update(str_audio_path.encode())

            name_id = hash_object.hexdigest()[:8]
            filename = f"{name_id}.npy"
            filepath = save_dir / filename
            np.save(filepath, activation.cpu().numpy())

            activation_paths.append(filepath)
            filtered_labels.append(label_encoder.transform(
                [Path(audio_path).parent.parent.name])[0])
    return activation_paths, filtered_labels


def train(train_loader, input_size, layer, device, num_epochs=10):
    """
    Train a model on a train dataset
    """
    model = EmotionCls(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()

        for X, y in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, test_loader, device):
    """
    Evaluates a model on a test dataset.
    Calculates accuracy and f1-score
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(
                test_loader, desc="Evaluation Progress"):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro")
    }

    return metrics


def plot_metrics(metrics_list, save_path):
    layers = [m[0] for m in metrics_list]
    accuracies = [m[1]["accuracy"] for m in metrics_list]
    f1_scores = [m[1]["f1_score"] for m in metrics_list]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(layers)+1), accuracies, color='b', label="Accuracy")
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy across layers")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(layers)+1), f1_scores, color='g', label="F1-score")
    plt.xlabel("Layers")
    plt.ylabel("F1-score")
    plt.title("F1-score across layers")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)


def save_metrics(metrics_list, save_path):
    """
    Saves computed metrics in .txt file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for layer, metrics in metrics_list:
            f.write(f"{layer}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=True,
        help="Path to wespeaker model pretrain_dir."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./dataset/train",
        help="Path to train audio files"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./dataset/test",
        help="Path to test audio files"
    )
    parser.add_argument(
        "--models_save_path",
        type=str,
        default="./models",
        help="Save path for trained models"
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./results/voxblink2_samresnet34/probing.txt",
        help="Save path for text result"
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./results/voxblink2_samresnet34/probing.png",
        help="Save path for visual result"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pretrain_dir):
        raise FileNotFoundError(f"Folder {args.pretrain_dir} does not exists.")
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")
    if not os.path.exists(args.models_save_path):
        raise FileNotFoundError(
            f"Folder {args.models_save_path} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)

    acts_model = GetActivations(model)

    train_files = get_audio_path(args.train_dir)
    test_files = get_audio_path(args.test_dir)

    acts = get_activations(acts_model, train_files[0], device)
    layers = [list(item.keys())[0] for item in acts["act"]]

    all_files = train_files + test_files    
    max_act_len = get_max_length(acts_model, all_files, device)

    metrics_list = []

    for layer in layers:
        train_acts, train_labels = get_activations_for_layer(
            acts_model, train_files, device, layer, max_act_len)
        test_acts, test_labels = get_activations_for_layer(
            acts_model, test_files, device, layer, max_act_len)

        train_dataset = ActivationDataset(train_acts, train_labels, max_act_len)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        test_dataset = ActivationDataset(test_acts, test_labels, max_act_len)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        sample_input, _ = train_dataset[0]
        input_size = sample_input.shape[0]
        
        trained_model = train(
            train_loader, input_size, layer, device)
        
        model_path = os.path.join(args.models_save_path, f"{layer}.pth")
        torch.save(trained_model.state_dict(), model_path)

        metrics = evaluate(trained_model, test_loader, device)
        metrics_list.append((layer, metrics))

        torch.cuda.empty_cache()

    save_metrics(metrics_list, args.text_save_path)
    plot_metrics(metrics_list, args.visual_save_path)


if __name__ == '__main__':
    main()
