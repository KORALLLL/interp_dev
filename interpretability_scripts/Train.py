import os
import shutil
from pathlib import Path
import random
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models import FormantPredictor
from datasets import DatasetFormant
from models.PhonemeTokenizer import PhonemeTokenizer
import wespeaker
from utils import extract_features
from models import MlpAdaLN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
PHONEME_VOCAB_PATH = r"C:\Users\Илья\Desktop\fignya\interp_dev\models\PhonemeTokenizer\phoneme_vocab.json"

tokenizer = PhonemeTokenizer(PHONEME_VOCAB_PATH)


class GetActivations(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.saved_out = None

    def save_identity(self, file_name):
        folder = Path("tmp_identity")
        folder.mkdir(exist_ok=True)
        torch.save(self.saved_out, folder / file_name)

    def delete_identity(self):
        if os.path.exists("tmp_identity"):
            shutil.rmtree("tmp_identity")

    def forward(self, x, target_layer, from_activation=False, identity_file=None):
        activations = {}
        model_front = self.model.model.front
        out = x

        if not from_activation:
            out = x.permute(0, 2, 1).unsqueeze(1)
            out = model_front.relu(model_front.bn1(model_front.conv1(out)))
            self.saved_out = out.clone()
            if identity_file:
                self.save_identity(identity_file)
            if target_layer == "first relu":
                activations["first relu"] = out
                return activations, out
        else:
            if identity_file and os.path.exists(f"tmp_identity/{identity_file}"):
                self.saved_out = torch.load(f"tmp_identity/{identity_file}", map_location=x.device)
            out = x

        for name, layer in model_front.named_children():
            if not name.startswith("layer"):
                continue
            for block_idx, block in layer.named_children():
                identity = self.saved_out

                out = block.relu(block.bn1(block.conv1(out)))
                if f"{name} relu 1" == target_layer:
                    activations[f"{name} relu 1"] = out
                    return activations, out

                out = block.bn2(block.conv2(out))
                out = block.SimAM(out)
                if f"{name} SimAM 1" == target_layer:
                    activations[f"{name} SimAM 1"] = out
                    return activations, out

                if block.downsample is not None:
                    identity = block.downsample(identity)
                out = block.relu(out + identity)
                self.saved_out = out.clone()
                if identity_file:
                    self.save_identity(identity_file)
                if f"{name} relu 2" == target_layer:
                    activations[f"{name} relu 2"] = out
                    return activations, out

        raise ValueError(f"Layer '{target_layer}' not found in model.")


def get_layers(model):
    layers = ["first relu"]
    for name, layer in model.model.front.named_children():
        if name.startswith("layer"):
            for _, _ in layer.named_children():
                layers.append(f"{name} relu 1")
                layers.append(f"{name} SimAM 1")
                layers.append(f"{name} relu 2")
    return layers


def get_activations(model, dataset_entries, device, chunk_num, layer):
    activations, tokens, masks, targets = [], [], [], []
    with torch.no_grad():
        for i, entry in enumerate(tqdm(dataset_entries, desc=f"Extracting {layer}")):
            feats = extract_features(entry["audio_path"], is_cut=False).unsqueeze(0).to(device)
            acts, _ = model(feats, layer, identity_file=f"identity_{chunk_num}_{i}.pt")

            activations.append(acts[layer].squeeze(0).cpu())

            token_ids = entry["phoneme_tokens"]
            att_mask = torch.ones_like(token_ids)

            tokens.append(token_ids)
            masks.append(att_mask)
            targets.append(entry["formants"])

    return activations, tokens, masks, targets



def pad_and_stack_feats(feats_list, hidden_dim=512):
    """
    feats_list: List of [C, H, W] тензоров
    Возвращает [B, C, H, max_W]
    """
    max_W = max(f.shape[2] for f in feats_list)
    C, H = feats_list[0].shape[0:2]

    padded_feats = []

    print(f"\n[DEBUG] Total embeddings: {len(feats_list)}")

    for idx, f in enumerate(feats_list):
        print(f"\n[Sample {idx}] Original shape: {f.shape}")

        if f.ndim != 3:
            raise ValueError(f"[Sample {idx}] Unsupported feature shape: {f.shape}, expected [C, H, W]")

        # Отладка для первых 3
        if idx < 3:
            mlp = MlpAdaLN(C=C, H=H, W=f.shape[2], hidden_dim=hidden_dim)
            params = mlp(f.unsqueeze(0))  # [1, C, H, W]

            names = ['alpha1', 'beta1', 'gamma1', 'alpha2', 'beta2', 'gamma2']
            for name, param in zip(names, params):
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

        # Паддинг по W
        pad_w = max_W - f.shape[2]
        if pad_w > 0:
            f = F.pad(f, (0, pad_w), value=0.0)

        padded_feats.append(f)

    return torch.stack(padded_feats)  # [B, C, H, max_W]



def train_model(loader, C, H, W, device, tokenizer):
    model = FormantPredictor(
        vocab_size=len(tokenizer.vocab),
        C=C,
        H=H,
        W=W,
        hidden_dim=512,
        num_formants=3,
        pad_token_id=tokenizer.pad_token_id,
        max_len=MAX_LEN,
        dropout=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    model.train()

    for _ in range(80):
        for feats, tokens, masks, tgts in loader:
            feats, tokens, masks, tgts = feats.to(device), tokens.to(device), masks.to(device), tgts.to(device)
            optimizer.zero_grad()
            preds = model(token_ids=tokens, attention_mask=masks, speech_embedding=feats)
            loss = criterion(preds, tgts)
            loss.backward()
            optimizer.step()

    return model


def evaluate(layer, y_pred, y_true):
    mse = nn.MSELoss()(torch.tensor(y_pred), torch.tensor(y_true))
    rmse = torch.sqrt(mse).item()
    return (layer, {"rmse": rmse})


def read_metrics(file_path):
    metrics_list = []
    current_layer = None
    current_metrics = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ':' not in line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                current_layer = line
                current_metrics = {}
            else:
                key, value = line.split(':', 1)
                current_metrics[key.strip()] = float(value.strip())

        if current_layer and current_metrics:
            metrics_list.append((current_layer, current_metrics))

    return metrics_list


def plot_rmse(metrics_list, save_path):
    layers = [m[0] for m in metrics_list]
    rmses = [m[1]["rmse"] for m in metrics_list]
    plt.figure(figsize=(10, 4))
    plt.plot(layers, rmses, marker='o', label="RMSE")
    plt.xticks(rotation=90)
    plt.ylabel("RMSE")
    plt.title("RMSE across layers")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot saved] {save_path}")


def main():
    AUDIO_DIR = r"C:\Users\Илья\Desktop\libritts\test-clean"
    CSV_DIR = r"C:\Users\Илья\Desktop\libritts\formants"
    WESPEAKER_DIR = r"C:\Users\Илья\Desktop\voxblink"
    METRICS_PATH = "metrics_main.txt"
    PLOT_PATH = "rmse_main.png"

    seen_layers = set()
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' not in line and line:
                    seen_layers.add(line)

    tokenizer = PhonemeTokenizer(PHONEME_VOCAB_PATH)
    dataset = DatasetFormant(CSV_DIR, AUDIO_DIR, tokenizer)
    model = wespeaker.load_model_local(WESPEAKER_DIR)
    model.set_device(DEVICE)
    acts_model = GetActivations(model)
    layers = [l for l in get_layers(model) if l != "pooling"]

    total_subset_size = 10
    chunk_size = 5

    all_indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(all_indices)
    selected_indices = all_indices[:total_subset_size]
    chunks = [selected_indices[i:i + chunk_size] for i in range(0, len(selected_indices), chunk_size)]

    for layer in layers:
        if layer in seen_layers:
            continue

        print(f"\n=== Processing layer: {layer} ===")
        all_feats, all_toks, all_masks, all_tgts = [], [], [], []

        for chunk_num, chunk_idx in enumerate(chunks):
            print(f"[Chunk {chunk_num}]")
            chunk_entries = [dataset[i] for i in chunk_idx]

            acts, tokens, masks, tgts = get_activations(
                acts_model, chunk_entries, DEVICE, chunk_num, layer
            )

            if len(acts) == 0:
                print(f"[Chunk {chunk_num}] — пропущен (нет валидных данных)")
                continue

            all_feats.extend(acts)
            all_toks.extend(tokens)
            all_masks.extend(masks)
            all_tgts.extend(tgts)

            gc.collect()
            torch.cuda.empty_cache()

        feats = pad_and_stack_feats(all_feats)  # [B, C, H, W]
        C, H, W = feats.shape[1:]

        max_tok_len = max(t.shape[0] for t in all_toks)
        toks = torch.stack([F.pad(t, (0, max_tok_len - t.shape[0]), value=tokenizer.pad_token_id) for t in all_toks])
        masks = torch.stack([F.pad(m, (0, max_tok_len - m.shape[0]), value=0) for m in all_masks])
        max_tgt_len = max(t.shape[0] for t in all_tgts)
        tgts = torch.stack([F.pad(t, (0, 0, 0, max_tgt_len - t.shape[0])) for t in all_tgts])

        loader = DataLoader(
            list(zip(feats, toks, masks, tgts)), batch_size=32, shuffle=True
        )

        model = train_model(loader, C=C, H=H, W=W, device=DEVICE, tokenizer=tokenizer)

        model.eval()
        preds = []
        with torch.no_grad():
            for X, T, M, Y in DataLoader(list(zip(feats, toks, masks, tgts)), batch_size=32):
                X, T, M = X.to(DEVICE), T.to(DEVICE), M.to(DEVICE)
                pred = model(token_ids=T, attention_mask=M, speech_embedding=X)
                preds.extend(pred.cpu().numpy())

        metric = evaluate(layer, preds, tgts.numpy())
        print(f"Metric for {layer}: {metric[1]}")

        with open(METRICS_PATH, 'a') as f:
            f.write(f"{layer}\n")
            for k, v in metric[1].items():
                f.write(f"{k}: {v}\n")

        seen_layers.add(layer)

        gc.collect()
        torch.cuda.empty_cache()

    acts_model.delete_identity()
    metrics = read_metrics(METRICS_PATH)
    plot_rmse(metrics, PLOT_PATH)
    os.remove(METRICS_PATH)


if __name__ == "__main__":
    main()
