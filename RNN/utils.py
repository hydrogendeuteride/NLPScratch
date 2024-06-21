import re
import pickle


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

    word_count['<UNKNOWN>'] = 0
    word_count['<PAD>'] = 0
    return pos_count, word_count


def build_vocab(word_counts, pos_counts):
    word_to_index = {'<PAD>': 0, '<UNKNOWN>': 1}
    tag_to_index = {}

    for i, word in enumerate(word_counts.keys(), 2):
        word_to_index[word] = i

    for i, tag in enumerate(pos_counts.keys()):
        tag_to_index[tag] = i

    return word_to_index, tag_to_index


def text_to_indices(processed_data, word_to_index, tag_to_index):
    X = []
    Y = []

    for sentence in processed_data:
        sentence_X = []
        sentence_Y = []

        for word, tag in sentence:
            word_idx = word_to_index.get(word, word_to_index.get('<UNKNOWN>'))
            tag_idx = tag_to_index[tag]

            sentence_X.append(word_idx)
            sentence_Y.append(tag_idx)

        X.append(sentence_X)
        Y.append(sentence_Y)

    return X, Y


def line_to_indices(sentence, word_to_index):
    sentence_x = []
    for word in sentence:
        word_idx = word_to_index.get(word, word_to_index.get('<UNKNOWN>'))
        sentence_x.append(word_idx)

    return sentence_x


def build_reverse_tag_index(tag_to_index):
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}
    return index_to_tag


def indices_to_tags(indices, index_to_tag):
    tags = [index_to_tag.get(index, '<UNKNOWN>') for index in indices]
    return tags


def save_data(vocab, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_and_save_vocab(data_file, vocab_file, line_num):
    data_line = read_file_to_list(data_file)
    processed_data_line = reader(data_line[:line_num])
    pos_cnt, word_cnt = count_word_POS(processed_data_line)

    word_to_idx, tag_to_idx = build_vocab(word_cnt, pos_cnt)
    idx_to_tag = build_reverse_tag_index(tag_to_idx)

    data_to_save = {
        'word_to_idx': word_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
        'word_count': len(word_cnt),
        'pos_count': len(pos_cnt)
    }

    save_data(data_to_save, vocab_file)
