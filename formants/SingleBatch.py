import os
import torch
from torch.utils.data import DataLoader
from tokenizer.phoneme_tokenizer import PhonemeTokenizer
from transformer.formant_predictor import FormantPredictor
from Dataset import DatasetFormant
from collate import my_collate_fn
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHONEME_VOCAB_PATH = r"C:\Users\Илья\Desktop\interp_dev\formants\tokenizer\phoneme_vocab.json"
CSV_DIR = r"C:\Users\Илья\Desktop\libritts\formants"
AUDIO_DIR = r"C:\Users\Илья\Desktop\libritts\test-clean"
EMBEDDING_DIR = r"C:\Users\Илья\Desktop\libritts\embeddings"

f1_min, f1_max = 100, 2000
f2_min, f2_max = 400, 3500
f3_min, f3_max = 1000, 4500

wandb.init(project="formant_predictor_Final", name="SEOT_single_batch_attention_mask", config={
    "batch_size": 64,
    "lr": 1e-3,
    "hidden_dim": 512,
    "max_len": 256,
    "steps": 300
})

class DatasetFormantFixed(DatasetFormant):
    def __init__(self, *args, fixed_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_index = fixed_index

    def __getitem__(self, idx):
        return super().__getitem__(self.fixed_index)

    def __len__(self):
        return 10000

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


all_csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
tokenizer = PhonemeTokenizer(PHONEME_VOCAB_PATH)

fixed_dataset = DatasetFormantFixed(CSV_DIR, AUDIO_DIR, tokenizer, all_csv_files, EMBEDDING_DIR, fixed_index=0)
train_loader = DataLoader(fixed_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=my_collate_fn)

model = FormantPredictor(
    vocab_size=len(tokenizer.vocab),
    hidden_dim=wandb.config.hidden_dim,
    num_formants=3,
    pad_token_id=tokenizer.pad_token_id,
    max_len=wandb.config.max_len,
    dropout=0.1
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)


print(">>> Single Example Overfit Test started")
for step, batch in enumerate(train_loader):
    model.train()
    phoneme_tokens = batch['phoneme_tokens'].to(DEVICE)
    formants_gt_norm = batch['formants'].to(DEVICE)
    attention_mask_float = (phoneme_tokens != tokenizer.pad_token_id).float().to(DEVICE)
    attention_mask_bool = (phoneme_tokens != tokenizer.pad_token_id).to(DEVICE)
    speech_embeddings = batch['speech_embedding'].to(DEVICE)

    pred_formants_norm = model(phoneme_tokens, speech_embeddings, attention_mask=attention_mask_bool)
    loss_mask = attention_mask_float.unsqueeze(-1)
    loss = (((pred_formants_norm - formants_gt_norm) ** 2) * loss_mask).sum() / loss_mask.sum()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    pred_formants_hz = denormalize_formants(pred_formants_norm)
    formants_gt_hz = denormalize_formants(formants_gt_norm)
    rmse_hz_value = rmse(pred_formants_hz, formants_gt_hz, attention_mask_float)


    wandb.log({
        "SEOT/loss_norm": loss.item(),
        "SEOT/rmse_hz": rmse_hz_value.item(),
        "SEOT/step": step
    })

    if step % 50 == 0 or step == wandb.config.steps - 1:
        print(f"[Step {step}] Loss (norm): {loss.item():.6f} | RMSE (Hz): {rmse_hz_value.item():.2f}")
        print(f"GT Formants (Hz): {formants_gt_hz[0, :10].cpu().numpy()}")
        print(f"Pred Formants (Hz): {pred_formants_hz[0, :10].detach().cpu().numpy()}")

    if step == wandb.config.steps - 1:
        break

print(">>> Finished")
