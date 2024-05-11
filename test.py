from hmm import HMMSupervised

def evaluate_accuracy(model, test_file):
    correct_tags = 0
    total_tags = 0

    test_data = model.reader(read_file_to_list(test_file))

    for sentence in test_data:
        #print(sentence)
        tokens = [token.rsplit('/')[0] for token in sentence[1:-1]]  
        true_tags = [token.rsplit('/')[1] for token in sentence[1:-1]]  
        predicted_tags = model.viterbi(tokens) 
        print(true_tags)
        print(predicted_tags)

        total_tags += len(true_tags)
        correct_tags += sum(1 for pred_tag, true_tag in zip(predicted_tags, true_tags) if pred_tag == true_tag)
        #print(total_tags, correct_tags)

    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    return accuracy

def read_file_to_list(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines

HMM = HMMSupervised()
HMM.load_probabilities('HMM_addingone_model.dat')

accuracy = evaluate_accuracy(HMM, 'tagged_test.txt')

print(f"Accuracy: {accuracy}")