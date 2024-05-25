from RNN_train import *
from train import *
from utils import *

data_line = read_file_to_list('../dataset/tagged_train_mini.txt')
processed_data_line = reader(data_line)
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
idx_to_tag = build_reverse_tag_index(tag_to_idx)

model = RNN(word_dim=len(word_cnt), tag_dim=len(pos_cnt), hidden_dim=100, bptt_truncate=4)
model.sgd_step(x1[3], y1[3], 0.005)

train_with_sgd(model, x1, y1, nepoch=10, evaluation_loss_after=1)

tag_i = model.predict(x1[3])
pos_tags = indices_to_tags(tag_i, idx_to_tag)
print(pos_tags)

model.save('../weight/save.txt')
