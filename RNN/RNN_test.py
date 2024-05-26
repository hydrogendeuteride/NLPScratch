from HMM.hmm import replace_quotes, custom_split
from RNN import *
from utils import *
import argparse


def rnn_tagger_test(model, input_file=None, output_file=None):
    if input_file and output_file:
        with open(input_file, 'r', encoding='utf-8') as file:
            sentences = file.read().strip().split('\n')
        results = [model.predict(replace_quotes(custom_split(sentence))) for sentence in sentences]
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
        results = [model.predict(replace_quotes(custom_split(sentence))) for sentence in sentences]
        for sentence, tags in zip(sentences, results):
            formatted_sentence = ' '.join(
                f"{word}/{tag}" for word, tag in zip(replace_quotes(custom_split(sentence)), tags))
            print(f"{formatted_sentence}\n")


def main():
    parser = argparse.ArgumentParser(description="HMM Batch Tagger")
    parser.add_argument('model_file', help='Model file path')
    parser.add_argument('--input_file', help='Input file path')
    parser.add_argument('--output_file', help='Output file path')

    args = parser.parse_args()

    data_line = read_file_to_list('../dataset/tagged_train_mini.txt')
    processed_data_line = reader(data_line)
    pos_cnt, word_cnt = count_word_POS(processed_data_line)
    word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

    x1, y1 = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
    idx_to_tag = build_reverse_tag_index(tag_to_idx)

    model = RNN(word_dim=len(word_cnt), tag_dim=len(pos_cnt), hidden_dim=100, bptt_truncate=4,
                params_path=args.model_file)

    rnn_tagger_test(model, input_file=args.input_file, output_file=args.output_file)
