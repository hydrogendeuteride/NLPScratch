from transformer import *
from train import *
from utils import *
import time
import numpy as np

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
            X.append(pad_sequence(indices[:i], max_len))
            Y.append(indices[i])

    X = pad_sequence(X, max_len)
    # Y = np.array(Y)
    return X, Y


data_line = read_file_to_list('../dataset/tagged_train.txt')
processed_data_line = reader(data_line[:100])
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

max_len = 320

X_train, Y_train = generate_sequence_data(processed_data_line, word_to_idx, max_len)
model = Transformer(vocab_size=len(word_to_idx),embed_dim=512, num_heads=8, ff_dim=2048, num_layers=2, max_len=max_len)
model.sgd_step(X_train[3], Y_train[3])

train_with_sgd(model, X_train, Y_train, nepoch=10, evaluation_loss_after=1)

# print(X_train[10], Y_train[10])
# print(len(X_train[10]))
# print(word_to_idx)
# print(word_cnt['<PAD>'])
# t = pad_sequence(X_train[10], 50)
# print(t)
# print(len(t))

# max_line, max_line_idx = 0, 0
# for i in range(len(processed_data_line)):
#     max_line = max(max_line, len(processed_data_line[i]))
#     if max_line == len(processed_data_line[i]):
#         max_line_idx = i
#
# print(max_line)
# print(max_line_idx)
# print(processed_data_line[max_line_idx])
# max = 284 -> token=320
