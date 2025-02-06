import argparse
import os

import chromadb
import numpy as np
import torch
import wespeaker


def extract_embeddings(audio_dir, device, pretrain_dir):
    model = wespeaker.load_model_local(pretrain_dir)
    model.set_device(device)

    embeddings = []

    for class_name in os.listdir(audio_dir):
        class_path = os.path.join(audio_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_path.endswith(('.wav', '.mp3')):
                    embedding = model.extract_embedding(file_path)

                    embedding = embedding.cpu().numpy()
                    embeddings.append({
                        'embedding': embedding,
                        'label': class_name
                    })

    return embeddings


def save_to_npy(embeddings, save_dir):
    numpy_embs = np.array(embeddings)
    np.save(os.path.join(save_dir, "numpy_embs.npy"), numpy_embs)


def save_to_chromadb(embeddings, db_path, split):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="gender_embeddings")

    collection.add(
        ids=[f"{split}_{i}" for i in range(len(embeddings))],
        embeddings=[item['embedding'] for item in embeddings],
        metadatas=[{"label": item['label'], "split": split}
                   for item in embeddings]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./train_audio",
                        help="Path to both train male and female audio files.")
    parser.add_argument("--test_dir", type=str, default="./test_audio",
                        help="Path to both test male and female audio files.")
    parser.add_argument("--pretrain_dir", type=str, default="./pretrain_dir",
                        help="Path to wespeaker model pretrain_dir.")
    parser.add_argument("--output", type=str, required=True, choices=[
                        "npy", "chromadb"], help="Embeddings saving format: npy or chromadb.")
    parser.add_argument("--save_path", type=str, default="./embeddings",
                        help="Save path for calculated embeddings")
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Folder {args.train_dir} does not exists.")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Folder {args.test_dir} does not exists.")
    if len(os.listdir(args.train_dir)) > 2:
        raise ValueError(f"Folder {args.train_dir} must contain 2 subfolders: male and female."
                         f"Found {len(os.listdir(args.train_dir))} folders")
    if len(os.listdir(args.test_dir)) > 2:
        raise ValueError(f"Folder {args.test_dir} must contain 2 subfolders: male and female."
                         f"Found {len(os.listdir(args.test_dir))} folders")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings = extract_embeddings(
        args.train_dir, device, args.pretrain_dir)
    test_embeddings = extract_embeddings(
        args.test_dir, device, args.pretrain_dir)

    if args.output == "npy":
        os.makedirs(args.save_path, exist_ok=True)
        embeddings = [{"train": train_embeddings, "test": test_embeddings}]
        save_to_npy(embeddings, args.save_path)
    elif args.output == "chromadb":
        save_to_chromadb(train_embeddings, args.save_path, split="train")
        save_to_chromadb(train_embeddings, args.save_path, split="test")


if __name__ == '__main__':
    main()
