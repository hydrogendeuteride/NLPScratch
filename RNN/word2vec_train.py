from word2vec import *
from train import *
from utils import *
import time
import numpy as np


data_line = read_file_to_list('../dataset/tagged_train.txt')
processed_data_line = reader(data_line[:100])
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
idx_to_tag = build_reverse_tag_index(tag_to_idx)

print(len(word_to_idx))
model = SkipGram(len(word_to_idx), 128)

train_skipgram(model, x1, len(word_to_idx), evaluation_interval=1)
