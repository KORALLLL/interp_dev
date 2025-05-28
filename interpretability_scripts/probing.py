import argparse
import os
from utils import get_audio_path, prepare_chunks, save_tmp, delete_tmp
from utils import (
    GetActivations,
    get_activations,
    get_layers
)
from utils import (
    evaluate_probing,
    read_metrics,
    plot_metrics,
    save_metrics,
    save_to_csv
)
from datasets import ActivationDataset
from models import train_probing_model
from pathlib import Path
import json

import numpy as np
from torch.utils.data import DataLoader
import torch
import wespeaker


def check_paths(*paths):
    """
    Checks if the specified paths exist.
    If at least one path does not exist - throws an exception.
    """
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder {path} does not exists.")


def get_remaining_layers(layers, resume_layer, done_message, on_done=None):
    """
    Returns a list of layers that have not yet been processed,
    starting from resume_layer.

    Used to continue training from where you stopped.
    After all layers have been processed, the on_done function is called.
    """
    if resume_layer is None:
        return layers
    if resume_layer == layers[-1]:
        print(done_message)
        if on_done:
            on_done()
        return []
    try:
        layers = layers[layers.index(resume_layer) + 1:]
    except ValueError:
        raise ValueError(
            f"Resume layer {resume_layer} not found in model.")
    return layers


def load_or_extract_acts(i, chunk, acts_model, device, layer, mode):
    """
    Loads or retrieves activations and labels
    for the specified chunk and layer.

    If saved activations and labels exist, they are loaded from disk.
    Otherwise, the activations are retrieved from the model.
    """
    acts_path = f"{mode}/tmp_acts_{i}.pt"
    labels_path = f"{mode}/tmp_labels_{i}.pt"

    if not os.path.exists(acts_path):
        activations, labels = get_activations(
            acts_model, chunk, device, i, layer, mode)
    else:
        acts_list = torch.load(acts_path)
        labels = torch.load(labels_path)
        activations = []
        for num, act in enumerate(acts_list):
            with torch.no_grad():
                act = act.to(device)
                layer_acts, _ = acts_model(
                    act,
                    layer,
                    True,
                    identity_file=f"{mode}_identity_{i}_{num}.pt")
                activations.append(layer_acts[layer])

    dataset = ActivationDataset(activations, labels, chunk)

    return dataset, activations, labels


def test(
        probing_model,
        skf_test,
        test_paths,
        test_labels,
        acts_model,
        device, layer, args
):
    """
    Tests the trained probing model
    on chunks of test data for the specified layer.
    """
    all_preds, all_labels, chunk_rows = [], [], []

    for i, (_, chunk_idx) in enumerate(
            skf_test.split(test_paths, test_labels)):
        chunk = test_paths[chunk_idx]
        dataset, activations, labels = load_or_extract_acts(i, chunk,
                                                            acts_model, device,
                                                            layer, "test")
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        save_tmp(activations, "test", f"tmp_acts_{i}.pt")
        save_tmp(labels, "test", f"tmp_labels_{i}.pt")

        probing_model.eval()
        y_pred_chunk, y_true_chunk, filenames = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, path_batch in loader:
                X_batch = X_batch.to(device)
                outputs = probing_model(X_batch).cpu()
                y_pred_chunk.extend(outputs.numpy())
                y_true_chunk.extend(y_batch.numpy())
                filenames.extend(
                    ['/'.join(Path(filepath).parts[-2:]) for filepath in
                     path_batch]
                )

        for filepath, true_label, pred in zip(filenames, y_true_chunk,
                                              y_pred_chunk):
            chunk_rows.append({
                "filename": str(filepath),
                "true_label": true_label,
                f"prediction_{layer}": int(pred > 0.5)
            })

        all_preds.extend(y_pred_chunk)
        all_labels.extend(y_true_chunk)

    return all_preds, all_labels, chunk_rows


def train_and_test(model, acts_model, train_files, test_files, device, args):
    """
    Trains and tests a probing model for
    activations of each layer of the SimAM ResNet model.

    Defines a list of model layers,
    then for each layer trains the probing model on training data,
    tests it on test data, saves metrics and predictions,
    saves training progress (layer) to a JSON file.
    """
    layers = get_layers(model)
    resume_file = "last_layer.json"
    message = "All layers have already been processed."

    if Path(resume_file).exists():
        with open(resume_file, "r") as f:
            resume_layer = json.load(f).get("last_layer")
    else:
        resume_layer = None

    layers = get_remaining_layers(layers, resume_layer, message)
    if not layers:
        return

    skf_train, train_paths, train_labels = prepare_chunks(
        train_files, args.chunk_size)
    skf_test, test_paths, test_labels = prepare_chunks(
        test_files, args.chunk_size)

    for layer in layers:
        print(f"Processing layer: {layer}")
        probing_model = None

        print("Training")
        for i, (_, chunk_idx) in enumerate(
            skf_train.split(train_paths, train_labels)
        ):
            chunk = train_paths[chunk_idx]
            dataset, activations, labels = load_or_extract_acts(
                i, chunk, acts_model, device, layer, "train")
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            probing_model = train_probing_model(
                loader,
                input_dim=dataset.audio_data.shape[-1],
                device=device,
                existing_model=probing_model
            )

            save_tmp(activations, "train", f"tmp_acts_{i}.pt")
            save_tmp(labels, "train", f"tmp_labels_{i}.pt")

        print("Testing")
        all_preds, all_labels, chunk_rows = test(
            probing_model,
            skf_test,
            test_paths,
            test_labels,
            acts_model,
            device,
            layer,
            args
        )

        metrics = evaluate_probing(
            layer, np.array(all_preds), np.array(all_labels))
        save_metrics([metrics], args.text_save_path)
        save_to_csv(chunk_rows, layer, args.csv_save_path)

        with open(resume_file, "w") as f:
            json.dump({"last_layer": layer}, f)

        torch.cuda.empty_cache()

    acts_model.delete_identity()
    delete_tmp("train")
    delete_tmp("test")
    plot_metrics(read_metrics(args.text_save_path), args.visual_save_path)


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
        default="./train_audio",
        help="Path to train audio files."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_audio",
        help="Path to test audio files."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of file chunks per epoch."
    )
    parser.add_argument(
        "--text_save_path",
        type=str,
        default="./result/probing.txt",
        help="Save path for text result."
    )
    parser.add_argument(
        "--csv_save_path",
        type=str,
        default="./result/probing.csv",
        help="Save path for csv result."
    )
    parser.add_argument(
        "--visual_save_path",
        type=str,
        default="./result/probing.png",
        help="Save path for visual result."
    )
    args = parser.parse_args()

    check_paths(args.pretrain_dir, args.train_dir, args.test_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wespeaker.load_model_local(args.pretrain_dir)
    model.set_device(device)
    acts_model = GetActivations(model)

    train_files = get_audio_path(args.train_dir)
    test_files = get_audio_path(args.test_dir)

    train_and_test(model, acts_model, train_files, test_files, device, args)


if __name__ == '__main__':
    main()
