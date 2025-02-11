#!/bin/bash

OUTPUT_FORMAT="npy"
SAVE_PATH="./embeddings"

python extract_embeddings.py --output "$OUTPUT_FORMAT" --save_path "$SAVE_PATH"

echo "Embeddings are saved in $SAVE_PATH."