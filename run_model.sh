#!/bin/bash

EMBS_SOURCE="npy"

python train_model.py --embeddings_source "$EMBS_SOURCE"

echo "Script has been executed."