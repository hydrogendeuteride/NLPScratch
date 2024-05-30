from RNN import *
from utils import *


def evaluate_rnn_accuracy(model, idx_to_tag, word_to_idx, test_file):
    correct_tags = 0
    total_tags = 0

    test_data = reader(read_file_to_list(test_file))

    for sentence in test_data:
        tokens = [token.rsplit('/')[0] for token in sentence[1:-1]]
        true_tags = [token.rsplit('/')[1] for token in sentence[1:-1]]
        word_indices = line_to_indices(tokens, word_to_idx)
        predicted_indices = model.predict(word_indices)
        predicted_tags = indices_to_tags(predicted_indices, idx_to_tag)
        print(true_tags)
        print(predicted_tags)

        total_tags += len(true_tags)
        correct_tags += sum(1 for pred_tag, true_tag in zip(predicted_tags, true_tags) if pred_tag == true_tag)

    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    return accuracy


loaded_data = load_data('../weight/vocab_data.pkl')

word_to_idx = loaded_data['word_to_idx']
tag_to_idx = loaded_data['tag_to_idx']
idx_to_tag = loaded_data['idx_to_tag']
word_count = loaded_data['word_count']
pos_count = loaded_data['pos_count']
