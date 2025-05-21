import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.mlp_adaln import MlpAdaLN
from models.transformer_for_linear_probing import FormantPredictor
from utils.extract_features import extract_features
from datasets.Dataset import DatasetFormant
from models.phoneme_tokenizer import PhonemeTokenizer
import wespeaker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256

f1_min, f1_max = 59.13345692616717, 2443.4098356318045
f2_min, f2_max = 318.60651180437134, 3601.0578907146446
f3_min, f3_max = 1010.6346934807632, 4596.379377720998

def denormalize_formants(norm_formants):
    f1 = norm_formants[..., 0] * (f1_max - f1_min) + f1_min
    f2 = norm_formants[..., 1] * (f2_max - f2_min) + f2_min
    f3 = norm_formants[..., 2] * (f3_max - f3_min) + f3_min
    return torch.stack([f1, f2, f3], dim=-1)

def rmse_hz(pred, target, mask):
    mse = ((pred - target) ** 2) * mask.unsqueeze(-1)
    mse_sum = mse.sum()
    count = mask.sum() * 3
    if count == 0:
        return torch.tensor(float("nan")).to(pred.device)
    return torch.sqrt(mse_sum / count)

def get_layers(model):
    layers = ["first relu"]
    model_front = model.model.front
    for name, layer in model_front.named_children():
        if name not in ["layer1", "layer2", "layer3", "layer4"]:
            continue
        c_relu = 0
        c_sim = 0
        for _, _ in layer.named_children():
            c_relu += 1
            layers.append(f"{name} relu {c_relu}")
            c_sim += 1
            layers.append(f"{name} SimAM {c_sim}")
            c_relu += 1
            layers.append(f"{name} relu {c_relu}")
    layers.append("pooling")
    return layers

def get_wespeaker_activation(audio_path, wespeaker_model, target_layer):
    feats = extract_features(audio_path)
    x = feats.transpose(0, 1).unsqueeze(0).unsqueeze(1)

    model_front = wespeaker_model.model.front
    out = model_front.relu(model_front.bn1(model_front.conv1(x)))

    if target_layer == "first relu":
        out = out.squeeze(0).mean(dim=-1).flatten()
        if out.shape[0] > 512:
            pool = nn.AdaptiveAvgPool1d(512)
            out = pool(out.unsqueeze(0)).squeeze(0)
        return out

    for name, layer in model_front.named_children():
        if name not in ["layer1", "layer2", "layer3", "layer4"]:
            continue

        c_relu = 0
        c_sim = 0

        for _, block in layer.named_children():
            identity = out

            c_relu += 1
            if f"{name} relu {c_relu}" == target_layer:
                out = block.relu(block.bn1(block.conv1(out)))
                return out.squeeze(0).mean(dim=-1).flatten()

            c_sim += 1
            if f"{name} SimAM {c_sim}" == target_layer:
                out = block.bn2(block.conv2(out))
                out = block.SimAM(out)
                return out.squeeze(0).mean(dim=-1).flatten()

            c_relu += 1
            if f"{name} relu {c_relu}" == target_layer:
                if block.downsample is not None:
                    identity = block.downsample(identity)
                out += identity
                out = block.relu(out)
                return out.squeeze(0).mean(dim=-1).flatten()

    if target_layer == "pooling":
        out = wespeaker_model.model.pooling(out)
        return out.squeeze(0)

    raise ValueError(f"Layer '{target_layer}' not found.")

def my_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    return {
        "phoneme_tokens": pad_sequence([x["phoneme_tokens"] for x in batch], batch_first=True),
        "formants": pad_sequence([x["formants"] for x in batch], batch_first=True),
        "audio_path": [x["audio_path"] for x in batch]
    }

def validate(model, adaln, loader, tokenizer, wespeaker_model, target_layer):
    model.eval()
    total_mse = 0
    total_rmse = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            lengths = (batch['phoneme_tokens'] != tokenizer.pad_token_id).sum(dim=1)
            mask = lengths <= MAX_LEN
            if mask.sum() == 0:
                continue

            phoneme_tokens = batch['phoneme_tokens'][mask].to(DEVICE)
            formants_gt = batch['formants'][mask].to(DEVICE)
            attention_mask = (phoneme_tokens != tokenizer.pad_token_id).to(DEVICE)
            audio_paths = [p for i, p in enumerate(batch['audio_path']) if mask[i]]

            phoneme_tokens = phoneme_tokens[:, :MAX_LEN]
            formants_gt = formants_gt[:, :MAX_LEN]
            attention_mask = attention_mask[:, :MAX_LEN]

            embeddings = [get_wespeaker_activation(p, wespeaker_model, target_layer) for p in audio_paths]
            embeddings = torch.stack(embeddings).to(DEVICE)

            alpha1, beta1, gamma1, alpha2, beta2, gamma2 = adaln(embeddings)
            adaln_params = (alpha1, beta1, gamma1, alpha2, beta2, gamma2)

            pred = model(token_ids=phoneme_tokens, attention_mask=attention_mask, adaln_params=adaln_params)

            mask_tensor = attention_mask.unsqueeze(-1).float()
            mse = (((pred - formants_gt) ** 2) * mask_tensor).sum() / mask_tensor.sum()

            pred_hz = denormalize_formants(pred)
            gt_hz = denormalize_formants(formants_gt)
            rmse = rmse_hz(pred_hz, gt_hz, attention_mask)

            total_mse += mse.item()
            total_rmse += rmse.item()
            count += 1

    return total_mse / count, total_rmse / count

def main():
    AUDIO_DIR = r"C:\Users\Илья\Desktop\libritts\test-clean"
    CSV_DIR = r"C:\Users\Илья\Desktop\libritts\formants"
    PHONEME_VOCAB_PATH = r"C:\Users\Илья\Desktop\interp_dev\models\phoneme_vocab.json"
    WESPEAKER_DIR = r"C:\Users\Илья\Desktop\voxblink"

    tokenizer = PhonemeTokenizer(PHONEME_VOCAB_PATH)
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    dataset = DatasetFormant(CSV_DIR, AUDIO_DIR, tokenizer, csv_files)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)

    wespeaker_model = wespeaker.load_model_local(WESPEAKER_DIR)
    wespeaker_model.set_device(DEVICE)
    wespeaker_model.model.eval()

    layer_names = get_layers(wespeaker_model)
    print("\n=== Training Mode ===")
    results = []

    for target_layer in layer_names:
        print(f"\n[Training on layer: {target_layer}]")

        model = FormantPredictor(
            vocab_size=len(tokenizer.vocab),
            hidden_dim=512,
            num_formants=3,
            pad_token_id=tokenizer.pad_token_id,
            max_len=MAX_LEN,
            dropout=0.1
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in tqdm(loader, desc=f"[{target_layer}] Epoch {epoch + 1}"):
                lengths = (batch['phoneme_tokens'] != tokenizer.pad_token_id).sum(dim=1)
                mask = lengths <= MAX_LEN
                if mask.sum() == 0:
                    continue

                phoneme_tokens = batch['phoneme_tokens'][mask].to(DEVICE)
                formants_gt = batch['formants'][mask].to(DEVICE)
                attention_mask = (phoneme_tokens != tokenizer.pad_token_id).to(DEVICE)
                audio_paths = [p for i, p in enumerate(batch['audio_path']) if mask[i]]

                phoneme_tokens = phoneme_tokens[:, :MAX_LEN]
                formants_gt = formants_gt[:, :MAX_LEN]
                attention_mask = attention_mask[:, :MAX_LEN]

                embeddings = [get_wespeaker_activation(p, wespeaker_model, target_layer) for p in audio_paths]
                embeddings = torch.stack(embeddings).to(DEVICE)

                adaln = MlpAdaLN(input_dim=embeddings.shape[-1], hidden_dim=512).to(DEVICE)
                alpha1, beta1, gamma1, alpha2, beta2, gamma2 = adaln(embeddings)
                adaln_params = (alpha1, beta1, gamma1, alpha2, beta2, gamma2)

                pred = model(token_ids=phoneme_tokens, attention_mask=attention_mask, adaln_params=adaln_params)

                mask_tensor = attention_mask.unsqueeze(-1).float()
                loss = (((pred - formants_gt) ** 2) * mask_tensor).sum() / mask_tensor.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_mse, avg_rmse = validate(model, adaln, loader, tokenizer, wespeaker_model, target_layer)
        results.append((target_layer, avg_mse, avg_rmse))
        print(f"[{target_layer}] MSE: {avg_mse:.4f} | RMSE (Hz): {avg_rmse:.2f}")

    print("\n=== Layer-wise Results ===")
    for name, mse, rmse in results:
        print(f"{name:20}  MSE: {mse:.4f}  RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()