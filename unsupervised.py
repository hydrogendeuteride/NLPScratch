import numpy as np
import re
from scipy.special import logsumexp

class HMMSupervised:
    def __init__(self) -> None:
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


def logsumexp(x, axis = None):
    x_max = np.max(x, axis=axis, keepdims=True)
    if np.isneginf(x_max).all():
        return -np.inf
    else:
        return np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def forward(obs_seq, states, trans_prob, emit_prob):
    alpha = np.full((len(obs_seq) + 1, len(states)), -np.inf)
    alpha[0, states.index('<START>')] = 0

    for t in range(1, len(obs_seq) + 1):
        for j in range(len(states)):
            alpha[t, j] = logsumexp(alpha[t - 1] + trans_prob[:, j]) + emit_prob[j, obs_seq[t-1]]

    return alpha

def backward(obs_seq, states, trans_prob, emit_prob):
    beta = np.full((len(obs_seq) + 1, len(states)), -np.inf)
    beta[-1, states.index('<END>')] = 0
    
    for t in range(len(obs_seq) - 1, -1, -1):
        for i in range(len(states)):
            temp_values = beta[t + 1] + emit_prob[:, obs_seq[t]] + trans_prob[i, :]
            beta[t, i] = logsumexp(temp_values)

    return beta

def forward_backward(obs_seq, states, trans_prob, emit_prob, convergence_threshold=1e-6):
    prev_trans_prob = np.copy(trans_prob)
    prev_emit_prob = np.copy(emit_prob)
    iteration = 0

    while True:
        print(trans_prob)
        print(emit_prob)
        alpha = forward(obs_seq, states, trans_prob, emit_prob)
        beta = backward(obs_seq, states, trans_prob, emit_prob)

        xi = np.full((len(obs_seq) - 1, len(states), len(states)), np.inf)
        for t in range(len(obs_seq) - 1):
            denom = logsumexp(alpha[t, :, None] + trans_prob + emit_prob[:, obs_seq[t+1]] + beta[t+1, :])
            for i in range(len(states)):
                num = alpha[t, i] + trans_prob[i, :] + emit_prob[:, obs_seq[t+1]] + beta[t+1, :]
                xi[t, i, :] = num - denom

        gamma = (alpha + beta) - logsumexp(alpha + beta, axis=1)

        new_trans_prob = logsumexp(xi, axis=0) - logsumexp(gamma[:-1], axis=0)
        new_emit_prob = np.full_like(emit_prob, -np.inf)

        for i in range(len(states)):
            denom = logsumexp(gamma[:, i])
            for o in range(emit_prob.shape[1]):
                mask = obs_seq == o
                if np.any(mask):
                    new_emit_prob[i, o] = logsumexp(gamma[mask, i]) - denom
                else:
                    new_emit_prob[i, o] = -np.inf

        if np.allclose(new_trans_prob, prev_trans_prob, atol=convergence_threshold) and \
           np.allclose(new_emit_prob, prev_emit_prob, atol=convergence_threshold):
            break

        prev_trans_prob, prev_emit_prob = new_trans_prob, new_emit_prob
        trans_prob, emit_prob = new_trans_prob, new_emit_prob
        iteration += 1

    return trans_prob, emit_prob, iteration
    
states = ['A', 'B', '<START>', '<END>']
obs_seq = [0, 1, 0, 1]
#trans_prob = np.array([[0.7, 0.3, 0, 0], [0.4, 0.6, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 1]])
#emit_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0, 0], [0, 0]])

num_states = len(states)

trans_prob = np.full((num_states, num_states), -np.inf)

trans_prob[states.index('<START>'), states.index('A')] = np.log(0.5)
trans_prob[states.index('<START>'), states.index('B')] = np.log(0.5)
trans_prob[states.index('A'), states.index('A')] = np.log(0.5)
trans_prob[states.index('A'), states.index('B')] = np.log(0.3)
trans_prob[states.index('A'), states.index('<END>')] = np.log(0.2)
trans_prob[states.index('B'), states.index('A')] = np.log(0.3)
trans_prob[states.index('B'), states.index('B')] = np.log(0.5)
trans_prob[states.index('B'), states.index('<END>')] = np.log(0.2)

num_observations = len(set(obs_seq))

emit_prob = np.full((num_states, num_observations), -np.inf)

emit_prob[states.index('A'), 0] = np.log(0.7)
emit_prob[states.index('A'), 1] = np.log(0.3)
emit_prob[states.index('B'), 0] = np.log(0.7)
emit_prob[states.index('B'), 1] = np.log(0.3)

trans_prob, emit_prob, iterations = forward_backward(obs_seq, states, trans_prob, emit_prob)
print("Converged Transition Probabilities:\n", trans_prob)
print("Converged Emission Probabilities:\n", emit_prob)
print("Iterations:", iterations)