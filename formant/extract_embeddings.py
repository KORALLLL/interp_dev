import argparse
import os
from pathlib import Path

import chromadb
import numpy as np
import torch
import wespeaker
import pandas as pd

def get_audio_path(audio_dir):
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob('**/*.wav')) + list(
        audio_dir.glob('**/*.mp3'))

    return audio_files

def extract_embeddings(audio_files, device, pretrain_dir):
    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device(device)

    embeddings = []

    for file_path in audio_files:
        embedding = model.extract_embedding(str(file_path))

        embedding = embedding.cpu().numpy()
        embeddings.append({
            'file_path': str(file_path),
            'embedding': embedding
        })

    return embeddings

def assign_labels(embeddings, formants_file):
    formants_df = pd.read_csv(formants_file)
    for emb in embeddings:
        file_name = Path(emb['file_path']).name
        formant_rows = formants_df[formants_df['FileName'] == file_name]
        if not formant_rows.empty:
            emb['f1'] = float(formant_rows['F1'].iloc[0])
            emb['f2'] = float(formant_rows['F2'].iloc[0])
            emb['f3'] = float(formant_rows['F3'].iloc[0])
        else:
            emb['f1'] = 0.0
            emb['f2'] = 0.0
            emb['f3'] = 0.0

def save_to_chromadb(embeddings, db_path, split):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="intonation_embeddings")

    collection.add(
        ids=[f"{split}_{i}" for i in range(len(embeddings))],
        embeddings=[item['embedding'] for item in embeddings],
        metadatas=[{
            "file_path": item['file_path'],
            "f1": item['f1'],
            "f2": item['f2'],
            "f3": item['f3'],
            "split": split
        } for item in embeddings]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/libri_tts_r_data_test_clean/LibriTTS_R/test-clean",
                        help="Path to both train male and female audio files.")
    parser.add_argument("--test_dir", type=str, default="./data/libri_tts_r_data_test_clean/LibriTTS_R/test-clean",
                        help="Path to both test male and female audio files.")
    parser.add_argument("--formants_file", type=str, default="./data/formants_data.csv",
                        help="Path to formants_data.csv.")
    parser.add_argument("--pretrain_dir", type=str, default="./intonation_contour/voxblink2_samresnet34",
                        help="Path to wespeaker model pretrain_dir.")
    parser.add_argument("--save_path", type=str, default="./intonation_contour/embeddings",
                        help="Save path for calculated embeddings")
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_audio_files = get_audio_path(args.train_dir)
    test_audio_files = get_audio_path(args.test_dir)

    train_embeddings = extract_embeddings(train_audio_files, device,
                                          args.pretrain_dir)
    test_embeddings = extract_embeddings(test_audio_files, device,
                                         args.pretrain_dir)

    assign_labels(train_embeddings, args.formants_file)
    assign_labels(test_embeddings, args.formants_file)

    save_to_chromadb(train_embeddings, args.save_path, split="train")
    save_to_chromadb(test_embeddings, args.save_path, split="test")

if __name__ == '__main__':
    main()