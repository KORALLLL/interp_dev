#!/bin/bash

TRAIN_DIR="./train_audio"
TEST_DIR="./test_audio"
PRETRAIN_DIR="./pretrain_dir"
OUTPUT_FORMAT="npy"
SAVE_PATH="./embeddings"

python extract_embeddings.py --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" \
--pretrain_dir "$PRETRAIN_DIR" --output "$OUTPUT_FORMAT" --save_path "$SAVE_PATH"

echo "Embeddings are saved in $SAVE_PATH."