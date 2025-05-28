import os
from pathlib import Path
from .base_dataset import BaseDataset

import chromadb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch


class ClassificationEmbeddingsDataset(BaseDataset):
    def __init__(
            self,
            source_path,
            split,
            source_type,
            collection_name="gender_embeddings"
    ):
        super().__init__()
        self.lb = LabelEncoder()
        self.source_path = source_path
        self.split = split
        self.source_type = source_type
        self.collection_name = collection_name

        self.prepare_data()

    def prepare_data(self):
        """
        Loads and prepares audio and label data for the dataset
        depending on the type of data source.

        Depending on the value of the self.source_type attribute,
        the method selects the method of data loading:
            - If source_type is npy, the data is loaded from a .npy format file
                using the get_npy_embeddings method.
            - If source_type is chromadb, the data is loaded from
                the ChromaDB database using the get_chroma_embeddings method.
        """
        if self.source_type == "npy":
            audio_data, labels, filenames = self.get_npy_embeddings(
                self.source_path, self.split)
        elif self.source_type == "chromadb":
            audio_data, labels, filenames = self.get_chroma_embeddings(
                self.source_path, self.split, self.collection_name)
        else:
            raise ValueError(
                f"Invalid source type: {self.source_type}. "
                "Choose 'npy' or 'chromadb'.")

        self.audio_data = torch.tensor(audio_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.filenames = filenames

    def get_npy_embeddings(self, source_path, split):
        """
        Reads embeddings from a .npy file
        """
        source = np.load(os.path.join(
            source_path, "numpy_embs.npy"), allow_pickle=True)[0]
        data = source[split]
        embeddings = np.array([item['embedding'] for item in data])
        labels = self.lb.fit_transform([item['label'] for item in data])
        filenames = [
            '/'.join(Path(item['file_path']).parts[-2:])
            for item in data
        ]
        return embeddings, labels, filenames

    def get_chroma_embeddings(self, source_path, split, collection_name):
        """
        Reads embeddings from ChromaDB
        """
        client = chromadb.PersistentClient(path=source_path)
        collection = client.get_collection(name=collection_name)
        results = collection.get(where={"split": split}, include=[
                                 "embeddings", "metadatas"])
        embeddings = np.array(results['embeddings'], dtype=np.float32)
        labels = self.lb.fit_transform(
            [item['label'] for item in results['metadatas']])
        filenames = [
            '/'.join(Path(item['file_path']).parts[-2:])
            for item in results['metadatas']
        ]
        return embeddings, labels, filenames
