#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DATA_DIR="DATA"
VOCAB_FILE="vocab.txt"
COOCCUR_FILE="cooccur.bin"
COOCCUR_SHUFFLED_FILE="cooccur_shuffled.bin"
SAVE_FILE="vectors"
MIN_COUNT=5

# Note that glove uses Hogwild - thread scheduling makes the results random despite using same seed
#SHUFFLE_SEED=42
#GLOVE_SEED=43

# --- Pipeline ---
# echo "1. Building vocabulary..."
cargo run --release --bin vocab -- --min-count ${MIN_COUNT} < ${DATA_DIR}/text8 > ${VOCAB_FILE}

# echo "2. Building co-occurrence matrix..."
cargo run --release --bin cooccur -- --vocab-file ${VOCAB_FILE} < ${DATA_DIR}/text8 > ${COOCCUR_FILE}

# echo "3. Shuffling co-occurrence records with seed ${SHUFFLE_SEED}..."
cargo run --release --bin shuffle < ${COOCCUR_FILE} > ${COOCCUR_SHUFFLED_FILE}

#echo "4. Training GloVe model with seed ${GLOVE_SEED}..."
cargo run --release --bin glove -- \
    --input-file ${COOCCUR_SHUFFLED_FILE} \
    --vocab-file ${VOCAB_FILE} \
    --save-file ${SAVE_FILE}

#echo "5. Evaluating the model..."
cargo run --release --bin eval

echo "All Done."
