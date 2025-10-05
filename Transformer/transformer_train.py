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

try:
    import cupy

    array_library = cupy
except ImportError:
    array_library = np


def generate_sequence_data(processed_data, word_to_index, max_len):
    X = []
    Y = []

    for sentence in processed_data:
        indices = [word_to_index.get(word, word_to_index['<UNKNOWN>']) for word, tag in sentence]
        for i in range(1, len(indices)):
            padded_seq = pad_sequence(indices[:i], max_len)
            X.append(padded_seq)
            Y.append(indices[i])

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    return X, Y


DATASET_PATH = os.path.join(REPO_ROOT, 'dataset', 'tagged_train.txt')
WEIGHT_PATH = os.path.join(REPO_ROOT, 'weight', 'word2vec_all.pth')

data_line = read_file_to_list(DATASET_PATH)
processed_data_line = reader(data_line)
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

#####################################################################
word2vec_model = SkipGram(len(word_to_idx), 256)
model_path = WEIGHT_PATH
if os.path.exists(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    # Map legacy keys if needed
    if 'embeddings.weight' in state and 'input.weight' not in state:
        state['input.weight'] = state['embeddings.weight']
    try:
        res = word2vec_model.load_state_dict(state, strict=False)
        print("Loaded word2vec checkpoint.",
              "missing:", getattr(res, 'missing_keys', None),
              "unexpected:", getattr(res, 'unexpected_keys', None))
    except Exception:
        print("Loaded word2vec checkpoint with relaxed mapping.")

embeddings = word2vec_model.get_embeddings()
#####################################################################

max_len = 320

X_train, Y_train = generate_sequence_data(processed_data_line, word_to_idx, max_len)
model = Transformer(vocab_size=len(word_to_idx), embed_dim=256, num_heads=8, ff_dim=1024, num_layers=3,
                    max_len=max_len, embedding_weight=embeddings, use_gpu=True)

# Build reverse map for qualitative samples
idx_to_word = {idx: w for w, idx in word_to_idx.items()}

# Build a few sample prompts (as lists of indices) from the training data
sample_prompts = []
for sent in processed_data_line[:3]:
    ids = [word_to_idx.get(w, word_to_idx['<UNKNOWN>']) for w, t in sent]
    # take a short prefix to keep compute light
    sample_prompts.append(ids[: min(12, len(ids)-1)])

train_transformer(
    model,
    X_train,
    Y_train,
    learning_rate=0.0001,
    nepoch=10,
    evaluation_loss_after=1,
    batch_size=16,
    print_every=10,
    clip_grad_norm=1.0,
    idx_to_word=idx_to_word,
    sample_prompts=sample_prompts,
    topk=5,
)

model.save(os.path.join(REPO_ROOT, 'transformer_model.pth'))
