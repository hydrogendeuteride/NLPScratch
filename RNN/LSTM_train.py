from LSTM import *
from train import *
from utils import *
import time
import numpy as np

try:
    import cupy

    array_library = cupy
except ImportError:
    array_library = np

data_line = read_file_to_list('../dataset/tagged_train.txt')
processed_data_line = reader(data_line[:100])
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
idx_to_tag = build_reverse_tag_index(tag_to_idx)

model = LSTM(word_dim=len(word_cnt), tag_dim=len(pos_cnt), hidden_dim=100, use_gpu=False)

start_time = time.time()
model.sgd_step(x1[3], y1[3], 0.005)
end_time = time.time()
execution_time = end_time - start_time

print("Execution time in seconds: ", execution_time)

train_with_sgd(model, x1, y1, learning_rate=0.02, nepoch=10, evaluation_loss_after=1)

tag_i = model.predict(x1[3])

if isinstance(tag_i, array_library.ndarray):
    np = cupy.get_array_module(tag_i)
    pos_tags = indices_to_tags(np.asnumpy(tag_i), idx_to_tag)
    print(pos_tags)
else:
    pos_tags = indices_to_tags(tag_i, idx_to_tag)
    print(pos_tags)