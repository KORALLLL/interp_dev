#!/bin/bash

PYTHON_SCRIPT="extract_embeddings.py"

OUTPUT_FORMAT="npy"
SAVE_PATH="./embeddings"

python "$PYTHON_SCRIPT" --output "$OUTPUT_FORMAT" --save_path "$SAVE_PATH"

echo "Embeddings are saved in $SAVE_PATH."