from .base_dataset import BaseDataset
import torch


class ActivationDataset(BaseDataset):
    """
    Dataset for preparing model activations
    """
    def __init__(self, activations, labels, filenames):
        self.audio_data = self.prepare_data(activations)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.filenames = [str(path) for path in filenames]

    def prepare_data(self, activations):
        activations = [act.clone() for act in activations]
        max_len = max(act.shape[-1] for act in activations)

        for i in range(len(activations)):
            pad_size = max_len - activations[i].shape[-1]
            activations[i] = torch.nn.functional.pad(
                activations[i], (0, pad_size), value=0.0)
            if len(activations[i].shape) != 2:
                activations[i] = activations[i].view(
                    activations[i].size(0), -1)

        return torch.stack(activations).squeeze(1)
