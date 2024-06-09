from word2vec import *
from train import *
from utils import *
import time
import numpy

try:
    import cupy

    array_library = cupy
except ImportError:
    array_library = np


def find_nearest(word, embeddings, word_to_index, index_to_word, k=5):
    npy = cupy.get_array_module(embeddings) if 'cupy' in str(type(embeddings)) else numpy

    if word not in word_to_index:
        return "Word not in dictionary."

    vec = embeddings[:, word_to_index[word]]
    similarity = npy.dot(embeddings.T, vec)
    norms = npy.linalg.norm(embeddings, axis=0) * npy.linalg.norm(vec)
    similarity /= norms

    nearest = npy.argsort(-similarity)[1:k + 1]
    nearest_words = [index_to_word[int(idx)] for idx in nearest.flatten()]
    return nearest_words


def analogy(word_a, word_b, word_c, embeddings, word_to_index, index_to_word):
    npy = cupy.get_array_module(embeddings) if 'cupy' in str(type(embeddings)) else numpy

    # print(embeddings.shape)
    vec_a = embeddings[word_to_index[word_a]]
    vec_b = embeddings[word_to_index[word_b]]
    vec_c = embeddings[word_to_index[word_c]]
    vec_result = vec_b - vec_a + vec_c
    similarity = npy.dot(embeddings.T, vec_result)
    norms = npy.linalg.norm(embeddings, axis=0) * npy.linalg.norm(vec_result)
    similarity /= norms

    nearest = npy.argsort(-similarity)[0:5]
    nearest_words = [index_to_word[int(idx)] for idx in nearest.flatten()]
    return nearest_words


data_line = read_file_to_list('../dataset/tagged_train.txt')
processed_data_line = reader(data_line[:5000])
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
idx_to_tag = build_reverse_tag_index(tag_to_idx)
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(len(word_to_idx))
model = SkipGram(len(word_to_idx), 256, use_gpu=True)

train_skipgram(model, x1, len(word_to_idx), evaluation_interval=1)
model.load('../weight/word2vec.pkl')

model.save('../weight/word2vec.pkl')

print("\nTesting with nearest words:")
test_words = ['as', 'serious', 'justice']
for word in test_words:
    if word in word_to_idx:
        nearest = find_nearest(word, model.W1, word_to_idx, idx_to_word, k=5)
        print(f"Nearest to '{word}': {nearest}")
    else:
        print(f"'{word}' not found in the vocabulary.")

print("\nTesting with analogies:")
triplets = [('king', 'man', 'queen'), ('paris', 'france', 'london')]
for a, b, c in triplets:
    if a in word_to_idx and b in word_to_idx and c in word_to_idx:
        result = analogy(a, b, c, model.W1.T, word_to_idx, idx_to_word)
        print(f"'{a}' is to '{b}' as '{c}' is to {result}")
    else:
        print(f"Words '{a}', '{b}', or '{c}' not found in vocabulary.")
