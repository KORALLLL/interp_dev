#!/bin/bash

PYTHON_SCRIPT="train_model.py"

EMBS_SOURCE="npy"

python "$PYTHON_SCRIPT" --embeddings_source "$EMBS_SOURCE"

echo "Script has been executed."