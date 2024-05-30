from RNN import *
from utils import *


def evaluate_rnn_accuracy(model, test_file):
    correct_tags = 0
    total_tags = 0

    test_data = reader(read_file_to_list(test_file))

    for sentence in test_data:
        tokens = [token.rsplit('/')[0] for token in sentence[1:-1]]
        true_tags = [token.rsplit('/')[1] for token in sentence[1:-1]]
        predicted_tags = model.predict(tokens)
        print(true_tags)
        print(predicted_tags)

        total_tags += len(true_tags)
        correct_tags += sum(1 for pred_tag, true_tag in zip(predicted_tags, true_tags) if pred_tag == true_tag)

    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    return accuracy



data_line = read_file_to_list('dataset/tagged_train.txt')
processed_data_line = reader(data_line[:5000])
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
idx_to_tag = build_reverse_tag_index(tag_to_idx)

accuracy = evaluate_accuracy(HMM, 'tagged_test.txt')

print(f"Accuracy: {accuracy}")
