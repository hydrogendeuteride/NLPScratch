import os
import sys
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Ensure repo root is on sys.path so `utils` works regardless of CWD
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.utils import *

# Optional: allow TF32 for faster matmuls on Ampere+
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # For PyTorch>=2.0: increase matmul precision hint
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    except Exception:
        pass


def subsample_sentences(sentences, word_counts, t=1e-5):
    total_words = sum(word_counts.values())
    word_probs = {}
    for word, count in word_counts.items():
        freq = count / total_words
        keep_prob = (np.sqrt(freq / t) + 1) * (t / freq) if freq > t else 1.0
        word_probs[word] = min(keep_prob, 1.0)

    filtered = [
        [word for word in sent if random.random() < word_probs.get(word, 1.0)]
        for sent in sentences
    ]
    return [sent for sent in filtered if len(sent) >= 2]


def generate_skipgram_pairs(sentences, window_size=3, pad_idx=0, unk_idx=1):
    pairs = []
    for sentence in sentences:
        sentence_length = len(sentence)
        for i, target in enumerate(sentence):
            if target in (pad_idx, unk_idx):
                continue
            for j in range(max(0, i - window_size), min(sentence_length, i + window_size + 1)):
                if i != j:
                    context = sentence[j]
                    if context in (pad_idx, unk_idx):
                        continue
                    pairs.append((target, context))
    return pairs


class SkipGramDataset(Dataset):
    def __init__(self, data):
        # data: list of (target, context) index pairs
        self.targets, self.contexts = zip(*data) if len(data) > 0 else ([], [])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        context = torch.tensor(self.contexts[idx], dtype=torch.long)
        return target, context


class SkipGram(nn.Module):
    """
    Efficient Skip-gram with Negative Sampling (SGNS).
    - Two embedding tables: input (target) and output (context).
    - Loss: -log sigma(u_c^T v_t) - sum_k log sigma(-u_n^T v_t)
    - Returns mean loss for a batch.
    """
    def __init__(self, vocab_size, embedding_dim, neg_k=10, sparse=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.neg_k = neg_k
        self.sparse = sparse
        self.input = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        self.output = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.input.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        nn.init.zeros_(self.output.weight)

    def forward(self, targets, contexts, negatives=None):
        # targets, contexts: (B,) Long
        v = self.input(targets)   # (B,D)
        u_pos = self.output(contexts)  # (B,D)
        pos_score = (v * u_pos).sum(-1)  # (B)
        pos_loss = F.logsigmoid(pos_score)

        if negatives is not None and negatives.numel() > 0:
            # negatives: (B,K)
            u_neg = self.output(negatives)  # (B,K,D)
            # (B,K)
            neg_score = torch.einsum('bkd,bd->bk', u_neg, v)
            neg_loss = F.logsigmoid(-neg_score).sum(-1)
        else:
            neg_loss = 0.0

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        return self.input.weight.detach().cpu().numpy()

def find_nearest(word, embeddings, word_to_index, index_to_word, k=5):
    import numpy as _np
    if word not in word_to_index:
        return "Word not in dictionary."
    vec = embeddings[word_to_index[word]]
    sim = embeddings @ vec
    norms = _np.linalg.norm(embeddings, axis=1) * _np.linalg.norm(vec)
    sim = sim / (norms + 1e-12)
    nearest = sim.argsort()[-(k + 1):][::-1][1:]
    return [index_to_word.get(int(i), '<UNKNOWN>') for i in nearest]

def build_unigram_table(word_counts, vocab_size, power=0.75, device='cpu', pad_idx=0, unk_idx=1):
    freq = torch.zeros(vocab_size, dtype=torch.float32)
    for idx, c in word_counts.items():
        if 0 <= idx < vocab_size:
            freq[idx] = float(c)
    # avoid sampling PAD/UNK as negatives
    if 0 <= pad_idx < vocab_size:
        freq[pad_idx] = 0.0
    if 0 <= unk_idx < vocab_size:
        freq[unk_idx] = 0.0
    prob = freq.pow(power)
    s = prob.sum()
    if s > 0:
        prob = prob / s
    else:
        prob = torch.full_like(prob, 1.0 / vocab_size)
    return prob.to(device)


if __name__ == "__main__":
    # Resolve paths relative to the repo root
    DATASET_PATH = os.path.join(REPO_ROOT, 'dataset', 'tagged_train.txt')
    WEIGHT_PATH = os.path.join(REPO_ROOT, 'weight', 'word2vec_all.pth')

    data_line = read_file_to_list(DATASET_PATH)
    processed_data_line = reader(data_line)
    pos_cnt, word_cnt = count_word_POS(processed_data_line)
    word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

    x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_counts = Counter(word for sentence in x1 for word in sentence)
    filtered_x1 = subsample_sentences(x1, word_counts)
    skipgram_pairs = generate_skipgram_pairs(filtered_x1, window_size=3, pad_idx=0, unk_idx=1)

    vocab_size = len(word_to_idx)
    embedding_dim = 256  # good trade-off for downstream Transformer(256-dim)
    neg_k = 10
    batch_size = 1024 if device.type == 'cuda' else 256
    lr = 2e-3
    epochs = 10
    num_workers = min(4, os.cpu_count() or 0)

    model = SkipGram(vocab_size, embedding_dim, neg_k=neg_k, sparse=False).to(device)

    # Negative sampling distribution (unigram^0.75)
    neg_prob = build_unigram_table(word_counts, vocab_size, power=0.75, device=device, pad_idx=0, unk_idx=1)

    # Optimizer and AMP
    use_amp = device.type == 'cuda'
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = SkipGramDataset(skipgram_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=(device.type=='cuda'), drop_last=False)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        for target, context in dataloader:
            target = target.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)

            # Sample negatives per-target
            negatives = torch.multinomial(neg_prob, num_samples=neg_k * target.size(0), replacement=True)
            negatives = negatives.view(target.size(0), neg_k)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                loss = model(target, context, negatives)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_steps += 1

        print(f"Epoch {epoch:02d} | loss {total_loss / max(1,total_steps):.4f}")

    # Save embeddings
    os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f"Model saved -> {WEIGHT_PATH}")

    print("\nTesting with nearest words:")
    E = model.get_embeddings()
    test_words = ['as', 'serious', 'justice']
    for w in test_words:
        if w in word_to_idx:
            print(w, '->', find_nearest(w, E, word_to_idx, idx_to_word, k=5))
        else:
            print(w, 'not in vocabulary')
