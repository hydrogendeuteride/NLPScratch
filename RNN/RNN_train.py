from function import *
import re
import pathlib


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class RNN:
    def __init__(self, word_dim, tag_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.bptt_truncate = bptt_truncate

        self.E = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (word_dim, word_dim))

        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (tag_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward(self, x):
        T = len(x)

        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((T, self.tag_dim))

        for t in np.arange(T):
            x_t = self.E[:, x[t]]
            s[t] = np.tanh(self.U.dot(x_t) + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0

        for i in np.arange(len(y)):
            o, s = self.forward(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def backward(self, x, y):
        T = len(x)
        o, s = self.forward(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdE = np.zeros(self.E.shape)

        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o, s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                dLdE[:, x[bptt_step]] += self.U.T.dot(delta_t)
                delta_t = self.W.T.dot(delta_t) * (1 - (s[bptt_step] ** 2))

        return [dLdU, dLdV, dLdW, dLdE]

    def sgd_step(self, x, y, learning_rate=0.01):
        dLdU, dLdV, dLdW, dLdE = self.backward(x, y)

        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        self.E -= learning_rate * dLdE


def read_file_to_list(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        lines = [line.strip() for line in lines]
    return lines


def reader(data):
    processed_data = []
    for line in data:
        line = re.sub(r"^\S+::\d+\s+", "", line)
        words_with_tags = line.split()

        sentence = [('<START>', '<START>')] + \
                   [(wt.rsplit('/')[0], wt.rsplit('/')[1]) for wt in words_with_tags if '/' in wt] + \
                   [('<END>', '<END>')]

        processed_data.append(sentence)

    return processed_data


def count_word_POS(processed_data):
    pos_count = {}
    word_count = {}

    for sentence in processed_data:
        for word, tag in sentence:

            if tag in pos_count:
                pos_count[tag] += 1
            else:
                pos_count[tag] = 1

            # Count words
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    return pos_count, word_count


def build_vocab(word_counts, pos_counts):
    word_to_index = {word: i for i, word in enumerate(word_counts.keys())}
    tag_to_index = {tag: i for i, tag in enumerate(pos_counts.keys())}
    return word_to_index, tag_to_index


def text_to_indices(processed_data, word_to_index, tag_to_index):
    X = []
    Y = []

    for sentence in processed_data:
        sentence_X = []
        sentence_Y = []

        for word, tag in sentence:
            word_idx = word_to_index.get(word, word_to_index.get('<UNKNOWN>'))  # Fallback to '<UNKNOWN>'
            tag_idx = tag_to_index[tag]

            sentence_X.append(word_idx)
            sentence_Y.append(tag_idx)

        X.append(sentence_X)
        Y.append(sentence_Y)

    return X, Y


data_line = read_file_to_list('../dataset/tagged_train_mini.txt')
processed_data_line = reader(data_line)
pos_cnt, word_cnt = count_word_POS(processed_data_line)
word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)

# print(len(pos_cnt), ' ', len(word_cnt))

x, y = text_to_indices(processed_data_line, word_to_idx, tag_to_idx)
sample_sentence = x[3]

model = RNN(word_dim=len(word_cnt), tag_dim=len(pos_cnt), hidden_dim=100, bptt_truncate=4)
output, hidden_states = model.forward(sample_sentence)

print("Output Probabilities:\n", output)