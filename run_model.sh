#!/bin/bash

EMBS_SOURCE="npy"
SOURCE_PATH="./embeddings"
EVAL_PATH="./scores/gender.txt"
VISUAL_PATH="./result/gender.png"


python train_model.py --embeddings_source "$EMBS_SOURCE" --source_path "$SOURCE_PATH" \
--eval_path "$EVAL_PATH" --visual_path "$VISUAL_PATH"

echo "Script has been executed."