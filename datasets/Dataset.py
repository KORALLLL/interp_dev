import os
import pandas as pd
import torch
import numpy as np

# Диапазоны по анализу
f1_min, f1_max = 100, 2000
f2_min, f2_max = 400, 3500
f3_min, f3_max = 1000, 4500

class DatasetFormant(torch.utils.data.Dataset):
    def __init__(self, csv_dir, audio_dir, tokenizer, csv_files=None, embedding_dir=None):
        self.data = []
        self.embedding_dir = embedding_dir
        csv_list = csv_files if csv_files is not None else os.listdir(csv_dir)

        for csv_filename in csv_list:
            filename_base = os.path.splitext(csv_filename)[0]
            parts = filename_base.split('_')
            folder1 = parts[0]
            folder2 = parts[1]
            audio_filename = filename_base + '.wav'
            audio_path = os.path.join(audio_dir, folder1, folder2, audio_filename)
            embedding_path = os.path.join(embedding_dir, filename_base + ".npy")

            df = pd.read_csv(os.path.join(csv_dir, csv_filename))

            if df[['F1', 'F2', 'F3']].isna().any().any():
                continue

            phoneme_list = df['Phoneme'].tolist()
            phoneme_tokens = tokenizer.encode(phoneme_list)
            formants = df[['F1', 'F2', 'F3']].values.astype(float)

            norm_f1 = (formants[:, 0] - f1_min) / (f1_max - f1_min)
            norm_f2 = (formants[:, 1] - f2_min) / (f2_max - f2_min)
            norm_f3 = (formants[:, 2] - f3_min) / (f3_max - f3_min)

            norm_formants = np.stack([norm_f1, norm_f2, norm_f3], axis=-1)

            self.data.append({
                'audio_path': audio_path,
                'embedding_path': embedding_path,
                'phoneme_tokens': phoneme_tokens,
                'formants': norm_formants
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        speech_embedding = np.load(entry['embedding_path'])
        speech_embedding = torch.tensor(speech_embedding, dtype=torch.float32)

        phoneme_tokens = torch.tensor(entry['phoneme_tokens'], dtype=torch.long)
        formants = torch.tensor(entry['formants'], dtype=torch.float32)

        return {
            'audio_path': entry['audio_path'],
            'phoneme_tokens': phoneme_tokens,
            'formants': formants,
            'speech_embedding': speech_embedding
        }
