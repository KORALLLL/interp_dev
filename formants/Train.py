import os
import torch
from torch.utils.data import DataLoader
from collate import my_collate_fn
from Dataset import DatasetFormant
from tokenizer.phoneme_tokenizer import PhonemeTokenizer
from transformer.formant_predictor import FormantPredictor
import wandb
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHONEME_VOCAB_PATH = r"C:\Users\Илья\Desktop\interp_dev\formants\tokenizer\phoneme_vocab.json"
CSV_DIR = r"C:\Users\Илья\Desktop\libritts\formants"
AUDIO_DIR = r"C:\Users\Илья\Desktop\libritts\test-clean"
EMBEDDING_DIR = r"C:\Users\Илья\Desktop\libritts\embeddings"
SEED = 42
VAL_SPLIT = 0.2

f1_min, f1_max = 100, 2000
f2_min, f2_max = 400, 3500
f3_min, f3_max = 1000, 4500

wandb.init(project="formant_predictor_Final", name="full_train_with_attention_mask_scheduler", config={
    "batch_size": 64,
    "lr": 1e-4,
    "hidden_dim": 512,
    "max_len": 256,
    "epochs": 10
})

all_csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
random.seed(SEED)
random.shuffle(all_csv_files)
split_idx = int(len(all_csv_files) * (1 - VAL_SPLIT))
train_csv_files = all_csv_files[:split_idx]
val_csv_files = all_csv_files[split_idx:]

tokenizer = PhonemeTokenizer(PHONEME_VOCAB_PATH)

train_dataset = DatasetFormant(CSV_DIR, AUDIO_DIR, tokenizer, train_csv_files, EMBEDDING_DIR)
val_dataset = DatasetFormant(CSV_DIR, AUDIO_DIR, tokenizer, val_csv_files, EMBEDDING_DIR)

train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=my_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=my_collate_fn)

model = FormantPredictor(
    vocab_size=len(tokenizer.vocab),
    hidden_dim=wandb.config.hidden_dim,
    num_formants=3,
    pad_token_id=tokenizer.pad_token_id,
    max_len=wandb.config.max_len,
    dropout=0.1
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=wandb.config.epochs)

def denormalize_formants(norm_formants):
    f1 = norm_formants[..., 0] * (f1_max - f1_min) + f1_min
    f2 = norm_formants[..., 1] * (f2_max - f2_min) + f2_min
    f3 = norm_formants[..., 2] * (f3_max - f3_min) + f3_min
    return torch.stack([f1, f2, f3], dim=-1)

def rmse(pred, target, mask):
    mse = ((pred - target) ** 2) * mask.unsqueeze(-1)
    mse_sum = mse.sum()
    count = mask.sum() * 3
    if count == 0:
        return torch.tensor(float('nan')).to(pred.device)
    return torch.sqrt(mse_sum / count)

def evaluate(model, loader, desc):
    model.eval()
    total_loss = 0
    total_rmse_hz = 0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            phoneme_tokens = batch['phoneme_tokens'].to(DEVICE)
            formants_gt_norm = batch['formants'].to(DEVICE)
            attention_mask_bool = (phoneme_tokens != tokenizer.pad_token_id).to(DEVICE)
            attention_mask_float = attention_mask_bool.float()
            speech_embeddings = batch['speech_embedding'].to(DEVICE)

            pred_formants_norm = model(phoneme_tokens, speech_embeddings, attention_mask=attention_mask_bool)

            loss_mask = attention_mask_float.unsqueeze(-1)
            loss = (((pred_formants_norm - formants_gt_norm) ** 2) * loss_mask).sum() / loss_mask.sum()

            pred_formants_hz = denormalize_formants(pred_formants_norm)
            formants_gt_hz = denormalize_formants(formants_gt_norm)
            rmse_hz_value = rmse(pred_formants_hz, formants_gt_hz, attention_mask_float)

            total_loss += loss.item()
            total_rmse_hz += rmse_hz_value.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_rmse_hz = total_rmse_hz / total_batches
    return avg_loss, avg_rmse_hz

print(">>> Full training loop started")

for epoch in range(wandb.config.epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        phoneme_tokens = batch['phoneme_tokens'].to(DEVICE)
        formants_gt_norm = batch['formants'].to(DEVICE)
        attention_mask_bool = (phoneme_tokens != tokenizer.pad_token_id).to(DEVICE)
        attention_mask_float = attention_mask_bool.float()
        speech_embeddings = batch['speech_embedding'].to(DEVICE)

        pred_formants_norm = model(phoneme_tokens, speech_embeddings, attention_mask=attention_mask_bool)

        loss_mask = attention_mask_float.unsqueeze(-1)
        loss = (((pred_formants_norm - formants_gt_norm) ** 2) * loss_mask).sum() / loss_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    train_loss_avg, train_rmse_hz_avg = evaluate(model, train_loader, "[Train Evaluation]")
    val_loss_avg, val_rmse_hz_avg = evaluate(model, val_loader, "[Validation Evaluation]")

    wandb.log({
        "train_epoch/loss_norm": train_loss_avg,
        "train_epoch/rmse_hz": train_rmse_hz_avg,
        "val_epoch/loss_norm": val_loss_avg,
        "val_epoch/rmse_hz": val_rmse_hz_avg,
        "lr": scheduler.get_last_lr()[0]
    })

    print(f"[Epoch {epoch}] Train Loss (norm): {train_loss_avg:.4f} | Train RMSE (Hz): {train_rmse_hz_avg:.2f} | Val Loss (norm): {val_loss_avg:.4f} | Val RMSE (Hz): {val_rmse_hz_avg:.2f}")

print(">>> Training finished.")
