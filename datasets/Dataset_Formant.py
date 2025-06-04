import os
import pandas as pd
import torch
import numpy as np
from .base_dataset import BaseDataset

# Диапазоны по анализу
F1_min, F1_max = 59.13345692616717, 2443.4098356318045
F2_min, F2_max = 318.60651180437134, 3601.0578907146446
F3_min, F3_max = 1010.6346934807632, 4596.379377720998

class DatasetFormant(BaseDataset):
    def __init__(self, csv_dir, audio_dir, tokenizer, single_file=None, csv_files=None):
        super().__init__()
        self.csv_dir = csv_dir
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.single_file = single_file
        self.csv_files = csv_files
        self.prepare_data()

    def prepare_data(self):
        self.audio_data = []
        self.labels = []

        if self.single_file:
            csv_list = [self.single_file]
        else:
            csv_list = self.csv_files if self.csv_files is not None else os.listdir(self.csv_dir)

        for csv_filename in csv_list:
            filename_base = os.path.splitext(csv_filename)[0]
            parts = filename_base.split('_')
            if len(parts) < 2:
                continue
            folder1, folder2 = parts[0], parts[1]
            audio_filename = filename_base + '.wav'
            audio_path = os.path.join(self.audio_dir, folder1, folder2, audio_filename)

            df = pd.read_csv(os.path.join(self.csv_dir, csv_filename))

            if df[['F1', 'F2', 'F3']].isna().any().any():
                continue

            phoneme_list = df['Phoneme'].tolist()
            phoneme_tokens = self.tokenizer.encode(phoneme_list)
            formants = df[['F1', 'F2', 'F3']].values.astype(float)

            norm_f1 = (formants[:, 0] - F1_min) / (F1_max - F1_min)
            norm_f2 = (formants[:, 1] - F2_min) / (F2_max - F2_min)
            norm_f3 = (formants[:, 2] - F3_min) / (F3_max - F3_min)

            norm_formants = np.stack([norm_f1, norm_f2, norm_f3], axis=-1)

            self.audio_data.append(audio_path)
            self.labels.append((phoneme_tokens, norm_formants))

    def __getitem__(self, idx):
        audio_path = self.audio_data[idx]
        phoneme_tokens, formants = self.labels[idx]

        return {
            'audio_path': audio_path,
            'phoneme_tokens': torch.tensor(phoneme_tokens, dtype=torch.long),
            'formants': torch.tensor(formants, dtype=torch.float32)
        }
