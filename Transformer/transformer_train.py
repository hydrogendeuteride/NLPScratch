import os
import sys

import numpy as np
import torch

# Make repo root importable regardless of how this file is executed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Support both package and script execution
try:
    from .transformer import Transformer, pad_sequence
except ImportError:
    from transformer import Transformer, pad_sequence

from utils.train import train_transformer
from utils.utils import read_file_to_list, reader, count_word_POS, build_vocab
from SkipGram.torch_skipgram import SkipGram

#####################################################################
# Word-level Transformer training (default)
#
# Hyperparameters are intentionally defined here (instead of argparse)
# so they are easy to see/modify.
#####################################################################
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "tagged_train.txt")
WORD2VEC_PATH = os.path.join(REPO_ROOT, "weight", "word2vec_all.pth")
SAVE_PATH = os.path.join(REPO_ROOT, "transformer_model.pth")

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


def generate_sequence_data(processed_data, word_to_index, max_len):
    X = []
    Y = []

    for sentence in processed_data:
        indices = [word_to_index.get(word, word_to_index["<UNKNOWN>"]) for word, _tag in sentence]
        for i in range(1, len(indices)):
            padded_seq = pad_sequence(indices[:i], max_len)
            X.append(padded_seq)
            Y.append(indices[i])

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    return X, Y


def main():
    data_line = read_file_to_list(DATASET_PATH)
    processed_data_line = reader(data_line)
    pos_cnt, word_cnt = count_word_POS(processed_data_line)
    word_to_idx, _tag_to_idx = build_vocab(word_cnt, pos_cnt)

    # Always provide an embedding table (SkipGram init), and optionally load a checkpoint.
    word2vec_model = SkipGram(len(word_to_idx), EMBED_DIM)
    if os.path.exists(WORD2VEC_PATH):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(WORD2VEC_PATH, map_location=device)
        # Map legacy keys if needed
        if isinstance(state, dict) and "embeddings.weight" in state and "input.weight" not in state:
            state["input.weight"] = state["embeddings.weight"]
        try:
            res = word2vec_model.load_state_dict(state, strict=False)
            print(
                "Loaded word2vec checkpoint.",
                "missing:", getattr(res, "missing_keys", None),
                "unexpected:", getattr(res, "unexpected_keys", None),
            )
        except Exception:
            print("Failed to load word2vec checkpoint; using randomly initialized SkipGram embeddings.")
    else:
        print(f"word2vec checkpoint not found: {WORD2VEC_PATH} (using random SkipGram embeddings)")

    embeddings = word2vec_model.get_embeddings()

    X_train, Y_train = generate_sequence_data(processed_data_line, word_to_idx, MAX_LEN)
    model = Transformer(
        vocab_size=len(word_to_idx),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        embedding_weight=embeddings,
        use_gpu=True,
    )

    idx_to_word = {idx: w for w, idx in word_to_idx.items()}

    sample_prompts = []
    for sent in processed_data_line[:3]:
        ids = [word_to_idx.get(w, word_to_idx["<UNKNOWN>"]) for w, _t in sent]
        sample_prompts.append(ids[: min(12, len(ids) - 1)])

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
