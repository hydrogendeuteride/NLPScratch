from HMM.hmm import replace_quotes, custom_split
from LSTM import *
from utils import *
import argparse


def lstm_tagger_test(model, idx_to_tag, word_to_idx, input_file=None, output_file=None):
    if input_file and output_file:
        with open(input_file, 'r', encoding='utf-8') as file:
            sentences = file.read().strip().split('\n')
        results = []
        for sentence in sentences:
            processed_sentence = replace_quotes(custom_split(sentence))
            word_idx = line_to_indices(processed_sentence, word_to_idx)
            results.append(indices_to_tags(model.predict(word_idx), idx_to_tag))

        with open(output_file, 'w', encoding='utf-8') as file:
            for sentence, tags in zip(sentences, results):
                formatted_sentence = ' '.join(
                    f"{word}/{tag}" for word, tag in zip(replace_quotes(custom_split(sentence)), tags))
                file.write(f"{formatted_sentence.rstrip()}\n")

    else:
        print("Enter sentences:")
        sentences = []
        while True:
            sentence = input()
            if sentence == "":
                break
            sentences.append(sentence)
        results = []
        for sentence in sentences:
            processed_sentence = replace_quotes(custom_split(sentence))
            word_idx = line_to_indices(processed_sentence, word_to_idx)
            results.append(indices_to_tags(model.predict(word_idx), idx_to_tag))

        for sentence, tags in zip(sentences, results):
            formatted_sentence = ' '.join(
                f"{word}/{tag}" for word, tag in zip(replace_quotes(custom_split(sentence)), tags))
            print(f"{formatted_sentence}\n")


def main():
    parser = argparse.ArgumentParser(description="RNN Batch Tagger")
    parser.add_argument('model_file', help='Model file path')
    parser.add_argument('--input_file', help='Input file path')
    parser.add_argument('--output_file', help='Output file path')

    args = parser.parse_args()

    # data_line = read_file_to_list('dataset/tagged_train.txt')
    # processed_data_line = reader(data_line[:100])
    # pos_cnt, word_cnt = count_word_POS(processed_data_line)
    # word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)
    #
    # idx_to_tag = build_reverse_tag_index(tag_to_idx)

    loaded_data = load_data('weight/vocab_data_lstm_f32_8k.pkl')

    word_to_idx = loaded_data['word_to_idx']
    tag_to_idx = loaded_data['tag_to_idx']
    idx_to_tag = loaded_data['idx_to_tag']
    word_count = loaded_data['word_count']
    pos_count = loaded_data['pos_count']

    model = LSTM(word_dim=word_count, word_embed_dim=8192, tag_dim=pos_count,
                 hidden_dim=256, max_norm=5, params_path=args.model_file)

    lstm_tagger_test(model, idx_to_tag, word_to_idx, input_file=args.input_file, output_file=args.output_file)


if __name__ == '__main__':
    main()
