from transformer import *
from utils.train import *
from utils.utils import *
import numpy as np
from SkipGram.torch_skipgram import SkipGram
import torch
import os

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

#####################################################################
word2vec_model = SkipGram(len(word_to_idx), 512)
model_path = '../weight/word2vec_100.pth'
if os.path.exists(model_path):
    word2vec_model.load_state_dict(torch.load(model_path))
    print("Model loaded for further training.")

embeddings = word2vec_model.get_embeddings()
#####################################################################

max_len = 320

X_train, Y_train = generate_sequence_data(processed_data_line, word_to_idx, max_len)
model = Transformer(vocab_size=len(word_to_idx), embed_dim=512, num_heads=8, ff_dim=2048, num_layers=3,
                    max_len=max_len, embedding_weight=embeddings, use_gpu=True)

model.sgd_step(X_train[3], Y_train[3])
train_with_sgd(model, X_train, Y_train, learning_rate=0.001, nepoch=10, evaluation_loss_after=1)
