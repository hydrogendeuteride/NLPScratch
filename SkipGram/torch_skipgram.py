import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from utils.utils import *


class TextProcessor:
    def __init__(self, min_count=5, window_size=5, neg_sample_size=5, subsample_t=1e-5):
        self.min_count = min_count
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        self.subsample_t = subsample_t

    def process(self, raw_sentences):
        word_counts = Counter(word for sent in raw_sentences for word in sent)
        self.word_counts = {w: c for w, c in word_counts.items() if c >= self.min_count}
        self.vocab = list(self.word_counts.keys())
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

        total_words = sum(self.word_counts.values())
        self.word_probs = {w: 1 - math.sqrt(self.subsample_t / (c / total_words))
                           for w, c in self.word_counts.items()}

        self.sentences = [
            [w for w in sent if w in self.word_counts and random.random() > self.word_probs[w]]
            for sent in raw_sentences
        ]
        return self.sentences


class SkipGramDataset(Dataset):
    def __init__(self, sentences, processor):
        self.processor = processor
        self.data = []

        for sentence in sentences:
            for i in range(len(sentence)):
                target = sentence[i]
                window = random.randint(1, processor.window_size)
                start = max(0, i - window)
                end = min(len(sentence), i + window + 1)
                context_words = [sentence[j] for j in range(start, end) if j != i]

                for context in context_words:
                    self.data.append((
                        self.processor.word2idx[target],
                        self.processor.word2idx[context]
                    ))

        word_counts = np.array(list(processor.word_counts.values()))
        self.neg_dist = torch.from_numpy(word_counts ** 0.75).float()
        self.neg_dist /= self.neg_dist.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        negatives = torch.multinomial(self.neg_dist, self.processor.neg_sample_size, True)
        return torch.LongTensor([target]), torch.LongTensor([context]), negatives


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.target_emb = nn.Embedding(vocab_size, emb_dim)
        self.context_emb = nn.Embedding(vocab_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        range_val = 0.5 / self.target_emb.weight.size(1)
        nn.init.uniform_(self.target_emb.weight, -range_val, range_val)
        nn.init.constant_(self.context_emb.weight, 0)

    def forward(self, target, context, negatives):
        # 양성 샘플 계산
        target_emb = self.target_emb(target).squeeze()
        context_emb = self.context_emb(context).squeeze()
        pos_score = torch.mul(target_emb, context_emb).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_emb = self.context_emb(negatives).neg()
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self):
        return self.target_emb.weight.data.cpu().numpy()


def train_model(sentences, device='cuda'):
    processor = TextProcessor(min_count=5, window_size=5)
    processed = processor.process(sentences)

    dataset = SkipGramDataset(processed, processor)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = SkipGramNS(len(processor.vocab), 300).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))

    for epoch in range(20):
        model.train()
        total_loss = 0

        for targets, contexts, negatives in dataloader:
            targets = targets.to(device)
            contexts = contexts.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            test_words = ['king', 'woman', 'quick']
            embeddings = model.get_embeddings()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            for word in test_words:
                if word in processor.word2idx:
                    idx = processor.word2idx[word]
                    sim = np.dot(embeddings, embeddings[idx])
                    nearest = np.argsort(-sim)[1:6]
                    print(f"{word}: {[processor.idx2word[i] for i in nearest]}")

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}')

    return model, processor


def get_similarity_matrix(embeddings, word2idx, top_n=10):
    norms = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / norms[:, None]
    sim_matrix = np.dot(normalized, normalized.T)

    similar_words = {}
    for word, idx in word2idx.items():
        nearest = np.argsort(-sim_matrix[idx])[1:top_n + 1]
        similar_words[word] = [(word2idx[i], sim_matrix[idx][i]) for i in nearest]

    return similar_words


if __name__ == "__main__":
    data_line = read_file_to_list('../dataset/tagged_train.txt')
    processed_data = reader(data_line)

    trained_model, processor = train_model(processed_data)

    embeddings = trained_model.get_embeddings()
    np.save('../weights/word2vec_embeddings.npy', embeddings)

    test_words = ['king', 'woman', 'justice']
    for word in test_words:
        if word in processor.word2idx:
            idx = processor.word2idx[word]
            sim = np.dot(embeddings, embeddings[idx])
            nearest = np.argsort(-sim)[1:6]
            print(f"Nearest to '{word}': {[processor.idx2word[i] for i in nearest]}")
