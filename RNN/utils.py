import re


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
            word_idx = word_to_index.get(word, word_to_index.get('<UNKNOWN>'))
            tag_idx = tag_to_index[tag]

            sentence_X.append(word_idx)
            sentence_Y.append(tag_idx)

        X.append(sentence_X)
        Y.append(sentence_Y)

    return X, Y


def build_reverse_tag_index(tag_to_index):
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}
    return index_to_tag


def indices_to_tags(indices, index_to_tag):
    tags = [index_to_tag.get(index, '<UNK>') for index in indices]
    return tags
