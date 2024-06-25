from LSTM import *
from utils.utils import *


def evaluate_rnn_accuracy(model, idx_to_tag, word_to_idx, test_file):
    correct_tags = 0
    total_tags = 0

    test_data = reader(read_file_to_list(test_file))

    for sentence in test_data:
        tokens = [token[0] for token in sentence]
        true_tags = [token[1] for token in sentence]
        word_indices = line_to_indices(tokens, word_to_idx)
        predicted_indices = model.predict(word_indices)
        predicted_tags = indices_to_tags(predicted_indices, idx_to_tag)
        print(true_tags)
        print(predicted_tags)

        total_tags += len(true_tags)
        correct_tags += sum(1 for pred_tag, true_tag in zip(predicted_tags, true_tags) if pred_tag == true_tag)

    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    print(f"accuracy: = {correct_tags} / {total_tags} =  {accuracy}")


loaded_data = load_data('../weight/vocab_data_lstm_f32_8k.pkl')

word_to_idx = loaded_data['word_to_idx']
tag_to_idx = loaded_data['tag_to_idx']
idx_to_tag = loaded_data['idx_to_tag']
word_count = loaded_data['word_count']
pos_count = loaded_data['pos_count']

model = LSTM(word_dim=word_count, word_embed_dim=8192, tag_dim=pos_count, hidden_dim=256,
             max_norm=5, params_path='../weight/lstm_model_8k.pkl')

acc = evaluate_rnn_accuracy(model, idx_to_tag, word_to_idx, test_file='../dataset/tagged_test.txt')
