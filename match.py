import numpy as np

from hmm import HMMSupervised as supervised
from unsupervised import HMMUnsupervised as unsupervised
from unsupervised import read_file_to_list

def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.where(p == 0, epsilon, p)
    q = np.where(q == 0, epsilon, q)
    return np.sum(p * np.log(p / q))

def calculate_distance(q_trans_unsup, q_trans_viterbi, q_emit_unsup, q_emit_viterbi, t, y, gamma1):
    kl_trans = kl_divergence(q_trans_unsup[t], q_trans_viterbi[y])
    
    kl_emit = kl_divergence(q_emit_unsup[t], q_emit_viterbi[y])
    
    distance = gamma1 * kl_trans + (1 - gamma1) * kl_emit
    return distance


def compute_all_distances(q_trans_unsup, q_trans_viterbi, q_emit_unsup, q_emit_viterbi, gamma1):
    num_states_unsup = q_trans_unsup.shape[0]
    num_states_viterbi = q_trans_viterbi.shape[0]
    distances = np.zeros((num_states_unsup, num_states_viterbi))

    for t in range(num_states_unsup):
        for y in range(num_states_viterbi):
            distances[t, y] = calculate_distance(q_trans_unsup, q_trans_viterbi, q_emit_unsup, q_emit_viterbi, t, y, gamma1)
    
    return distances

data = read_file_to_list('tagged_train.txt', 1000)

num_states = 10
unsup = unsupervised(num_states)
processed_data = unsup.reader(data)
epochs = 1
unsup.learn_sentences(epochs, processed_data)
unsup.save_results()

u_trans_prob = unsup.trans_prob
u_emit_prob = unsup.emit_prob

sup = supervised()
sup.load_probabilities('HMM_addingone_model.dat')

s_trans_prob = sup.transition_probs
s_emit_prob = sup.emission_probs

gamma1 = 0.5
q_trans_unsup = u_trans_prob
q_trans_viterbi = s_trans_prob
q_emit_unsup = u_emit_prob
q_emit_viterbi = s_emit_prob

distances = compute_all_distances(q_trans_unsup, q_trans_viterbi, q_emit_unsup, q_emit_viterbi, gamma1)
print("Distances between states:")
print(distances)