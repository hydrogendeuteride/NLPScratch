import os
from collections import deque, Counter
import random
from utils.utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cupy


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


def generate_skipgram_pairs(sentences, window_size=3):
    pairs = []
    for sentence in sentences:
        sentence_length = len(sentence)
        for i, target in enumerate(sentence):
            for j in range(max(0, i - window_size), min(sentence_length, i + window_size + 1)):
                if i != j:
                    context = sentence[j]
                    pairs.append((target, context))
    return pairs


class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.targets, self.contexts = zip(*data)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = torch.LongTensor([self.targets[idx]])
        context = torch.LongTensor([self.contexts[idx]])
        return target, context


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, max_norm=1.0)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, target):
        embed = self.embeddings(target)
        out = self.output_layer(embed)
        return out

    def get_embeddings(self):
        return self.embeddings.weight.data.cpu().numpy()

def find_nearest(word, embeddings, word_to_index, index_to_word, k=5):
    npy = cupy.get_array_module(embeddings) if 'cupy' in str(type(embeddings)) else np

    if word not in word_to_index:
        return "Word not in dictionary."

    vec = embeddings[word_to_index[word]]
    similarity = npy.dot(embeddings, vec)
    norms = npy.linalg.norm(embeddings, axis=1) * npy.linalg.norm(vec)
    similarity /= norms

    nearest = npy.argsort(-similarity)[1:k + 1]
    nearest_words = [index_to_word[int(idx)] for idx in nearest.flatten()]
    return nearest_words

if __name__ == "__main__":
    data_line = read_file_to_list('../dataset/tagged_train.txt')
    processed_data_line = reader(data_line)
    pos_cnt, word_cnt = count_word_POS(processed_data_line)
    word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

    x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
    idx_to_tag = build_reverse_tag_index(tag_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # skipgram_pairs = generate_skipgram_pairs(x1)

    word_counts = Counter(word for sentence in x1 for word in sentence)
    filtered_x1 = subsample_sentences(x1, word_counts)
    skipgram_pairs = generate_skipgram_pairs(filtered_x1)

    print("fin")

    embedding_dim = 256
    learning_rate = 0.003
    epochs = 50

    model = SkipGram(len(word_to_idx), embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model_path = '../weight/word2vec_all.pth'
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    #     print("Model loaded for further training.")

    dataset = SkipGramDataset(skipgram_pairs)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            target, context = data
            target, context = target.to(device), context.to(device)
            target = target.squeeze()
            context = context.squeeze()

            optimizer.zero_grad()
            output = model(target)
            loss = criterion(output, context)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}')

    model.eval()
    embeddings = model.embeddings.weight.data.cpu().numpy()

    torch.save(model.state_dict(), '../weight/word2vec_all.pth')
    print("Model saved.")

    print("\nTesting with nearest words:")
    embeddings = model.get_embeddings()
    print(embeddings.shape)
    test_words = ['as', 'serious', 'justice']
    for word in test_words:
        if word in word_to_idx:
            nearest = find_nearest(word, embeddings, word_to_idx, idx_to_word, k=5)
            print(f"Nearest to '{word}': {nearest}")
        else:
            print(f"'{word}' not found in the vocabulary.")
