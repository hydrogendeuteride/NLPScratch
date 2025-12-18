import os
import sys

import numpy as np

# Make repo root importable regardless of how this file is executed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Support both package and script execution
try:
    from .transformer import Transformer, pad_sequence
    from .bpe_encoder import BPEEncoder
except ImportError:
    from transformer import Transformer, pad_sequence
    from bpe_encoder import BPEEncoder

from utils.train import train_transformer
from utils.utils import read_file_to_list, reader

#####################################################################
# BPE Transformer training
#
# Hyperparameters are intentionally defined here (instead of argparse)
# so they are easy to see/modify.
#####################################################################
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "tagged_train.txt")
BPE_MODEL_PATH = os.path.join(REPO_ROOT, "weight", "bpe.pt")
RETRAIN_BPE = False

BPE_VOCAB_SIZE = 8000
BPE_MIN_FREQUENCY = 2

SAVE_PATH = os.path.join(REPO_ROOT, "transformer_model_bpe.pth")

MAX_LEN = 320
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 3

LEARNING_RATE = 0.0001
NEPOCH = 10
BATCH_SIZE = 16
EVALUATION_LOSS_AFTER = 1
PRINT_EVERY = 10
CLIP_GRAD_NORM = 1.0
TOPK = 5


def generate_bpe_sequence_data(token_seqs, bpe: BPEEncoder, max_len: int):
    X = []
    Y = []

    for tokens in token_seqs:
        ids = bpe.encode_ids(tokens, add_bos=True, add_eos=True)
        for i in range(1, len(ids)):
            padded_seq = pad_sequence(ids[:i], max_len, pad_token=bpe.pad_id)
            X.append(padded_seq)
            Y.append(ids[i])

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    return X, Y


def main():
    data_line = read_file_to_list(DATASET_PATH)
    processed_data_line = reader(data_line)

    # Train BPE on raw word tokens (drop <START>/<END> produced by reader()).
    token_seqs = [[w for (w, _t) in sent[1:-1]] for sent in processed_data_line]

    if (not RETRAIN_BPE) and os.path.exists(BPE_MODEL_PATH):
        bpe = BPEEncoder.load(BPE_MODEL_PATH)
        print(f"Loaded BPE model: {BPE_MODEL_PATH} (vocab_size={bpe.vocab_size})")
    else:
        bpe = BPEEncoder().train(
            token_seqs,
            vocab_size=BPE_VOCAB_SIZE,
            min_frequency=BPE_MIN_FREQUENCY,
        )
        os.makedirs(os.path.dirname(BPE_MODEL_PATH), exist_ok=True)
        bpe.save(BPE_MODEL_PATH)
        print(f"Saved BPE model: {BPE_MODEL_PATH} (vocab_size={bpe.vocab_size})")

    X_train, Y_train = generate_bpe_sequence_data(token_seqs, bpe, MAX_LEN)
    model = Transformer(
        vocab_size=bpe.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        embedding_weight=None,
        use_gpu=True,
    )

    idx_to_word = {int(i): t for i, t in bpe.id_to_token.items()}

    sample_prompts = []
    for tokens in token_seqs[:3]:
        ids = bpe.encode_ids(tokens, add_bos=True, add_eos=False)
        sample_prompts.append(ids[: min(12, max(1, len(ids) - 1))])

    train_transformer(
        model,
        X_train,
        Y_train,
        learning_rate=LEARNING_RATE,
        nepoch=NEPOCH,
        evaluation_loss_after=EVALUATION_LOSS_AFTER,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        clip_grad_norm=CLIP_GRAD_NORM,
        idx_to_word=idx_to_word,
        sample_prompts=sample_prompts,
        topk=TOPK,
    )

    model.save(SAVE_PATH)
    print(f"Saved transformer weights: {SAVE_PATH}")


if __name__ == "__main__":
    main()

