import argparse
import re
import ast
from collections import Counter
import math


def custom_split(sentence):
    tokens = re.findall(r'\w+|[,.!?;"]', sentence)
    return tokens


def replace_quotes(sentence):
    inside_quote = False
    for i, word in enumerate(sentence):
        if word == '"':
            if inside_quote:
                sentence[i] = "''"
                inside_quote = False
            else:
                sentence[i] = "``"
                inside_quote = True
    return sentence


class HMMSupervised:
    def __init__(self) -> None:
        self.tag_counts = Counter()
        self.word_tag_counts = Counter()
        self.tag_pair_counts = Counter()
        self.transition_probs = {}
        self.emission_probs = {}
        self.START = "<START>"
        self.STOP = "<STOP>"

    def reader(self, data):
        processed_data = []
        for line in data:
            line = re.sub(r"^\S+::\d+\s+", "", line)
            words_with_tags = line.split()

            processed_data.append([self.START] + words_with_tags + [self.STOP])
        return processed_data

    def count(self, data, save=False):
        for sentence in data:
            previous_tag = self.START

            for item in sentence[1:-1]:
                word, tag = item.rsplit('/', 1)
                self.word_tag_counts[(tag, word)] += 1
                self.tag_counts[tag] += 1
                self.tag_pair_counts[(previous_tag, tag)] += 1
                previous_tag = tag

            self.tag_pair_counts[(previous_tag, self.STOP)] += 1
            self.tag_counts[self.STOP] += 1

        if (save):
            with open('HMM_counts.dat', 'w', encoding='utf-8') as f:
                f.write(str(dict(self.tag_counts)) + "\n")
                f.write(str(dict(self.word_tag_counts)) + "\n")
                f.write(str(dict(self.tag_pair_counts)) + "\n")

    def load_counts(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.tag_counts = dict(ast.literal_eval(f.readline().strip()))
            self.word_tag_counts = dict(ast.literal_eval(f.readline().strip()))
            self.tag_pair_counts = dict(ast.literal_eval(f.readline().strip()))

    def calculate_probabilities(self, save=True):
        unique_tags = len(self.tag_counts)
        unique_words = len(set(word for _, word in self.word_tag_counts.keys()))

        self.transition_probs = {k: (v + 1) / (self.tag_counts[k[0]] + unique_tags)
                                 for k, v in self.tag_pair_counts.items()}

        self.emission_probs = {k: (v + 1) / (self.tag_counts[k[0]] + unique_words)
                               for k, v in self.word_tag_counts.items()}

        if (save):
            with open('HMM_addingone_model.dat', 'w', encoding='utf-8') as f:
                f.write(str(dict(self.tag_counts)) + "\n")
                f.write(str(self.transition_probs) + "\n")
                f.write(str(self.emission_probs) + "\n")

    def load_probabilities(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.tag_counts = dict(ast.literal_eval(f.readline().strip()))
            self.transition_probs = dict(ast.literal_eval(f.readline().strip()))
            self.emission_probs = dict(ast.literal_eval(f.readline().strip()))

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        START, STOP = '<START>', '<STOP>'

        for tag in self.tag_counts:
            V[0][tag] = math.log(self.transition_probs.get((START, tag), 1e-10)) + math.log(
                self.emission_probs.get((tag, sentence[0]), 1e-10))
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            newpath = {}

            for tag in self.tag_counts:
                (log_prob, state) = max((V[t - 1][prev] + math.log(self.transition_probs.get((prev, tag), 1e-10)) +
                                         math.log(self.emission_probs.get((tag, sentence[t]), 1e-10)), prev)
                                        for prev in self.tag_counts)

                V[t][tag] = log_prob
                newpath[tag] = path[state] + [tag]

            path = newpath

        (log_prob, state) = max((V[len(sentence) - 1][tag] +
                                 math.log(self.transition_probs.get((tag, STOP), 1e-10)), tag)
                                for tag in self.tag_counts)

        return path[state]

    def viterbi_tagger_test(self, model_path, input_file=None, output_file=None):
        self.load_probabilities(model_path)

        if input_file and output_file:
            with open(input_file, 'r', encoding='utf-8') as file:
                sentences = file.read().strip().split('\n')
            results = [self.viterbi(replace_quotes(custom_split(sentence))) for sentence in sentences]
            with open(output_file, 'w', encoding='utf-8') as file:
                for sentence, tags in zip(sentences, results):
                    formatted_sentence = ' '.join(
                        f"{word}/{tag}" for word, tag in zip(replace_quotes(custom_split(sentence)), tags))
                    file.write(f"{formatted_sentence.rstrip()}\n")

        else:
            print("Enter sentences :")
            sentences = []
            while True:
                sentence = input()
                if sentence == "":
                    break
                sentences.append(sentence)
            results = [self.viterbi(replace_quotes(custom_split(sentence))) for sentence in sentences]
            for sentence, tags in zip(sentences, results):
                formatted_sentence = ' '.join(
                    f"{word}/{tag}" for word, tag in zip(replace_quotes(custom_split(sentence)), tags))
                print(f"{formatted_sentence}\n")


def read_file_to_list(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        lines = [line.strip() for line in lines]
    return lines


# EXAMPLE:

# filename = 'nlp/tagged_train.txt'
# data = read_file_to_list(filename)
# HMM = HMMSupervised()
# tmp = HMM.reader(data)
# HMM.count(tmp, save=True)
# HMM.calculate_probabilities(save=True)
# HMM.load_probabilities('HMM_addingone_model.dat')
# HMM.viterbi_tagger_test('HMM_addingone_model.dat')

def main():
    parser = argparse.ArgumentParser(description="HMM Batch Tagger")
    parser.add_argument('model_file', help='Model file path')
    parser.add_argument('--input_file', help='Input file path')
    parser.add_argument('--output_file', help='Output file path')

    args = parser.parse_args()

    HMM = HMMSupervised()
    HMM.viterbi_tagger_test(args.model_file, args.input_file, args.output_file)


if __name__ == "__main__":
    main()
