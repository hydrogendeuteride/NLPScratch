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

def forward(obs_seq, states, start_prob, trans_prob, emit_prob):
    alpha = np.full((len(obs_seq), len(states)), -np.inf)
    alpha[0, :] = start_prob + emit_prob[:, obs_seq[0]]
    for t in range(1, len(obs_seq)):
        for j in range(len(states)):
            alpha[t, j] = logsumexp(alpha[t - 1] + trans_prob[:, j]) + emit_prob[j, obs_seq[t]]

    return alpha

def backward(obs_seq, states, trans_prob, emit_prob):
    beta = np.full((len(obs_seq), len(states)), -np.inf)
    beta[-1, :] = 0
    for t in range(len(obs_seq) - 2, -1, -1):
        for i in range(len(states)):
            beta[t, i] = logsumexp(beta[t + 1] + trans_prob[i, :] + emit_prob[:, obs_seq[t + 1]])

    return beta

def forward_backward(obs_seq, states, start_prob, trans_prob, emit_prob, convergence_threshold=1e-6):
    prev_trans_prob = np.copy(trans_prob)
    prev_emit_prob = np.copy(emit_prob)
    iteration = 0

    while True:
        alpha = forward(obs_seq, states, start_prob, trans_prob, emit_prob)
        beta = backward(obs_seq, states, trans_prob, emit_prob)
        print(trans_prob)
        print(emit_prob)
        print(iteration)

        xi = np.full((len(obs_seq) - 1, len(states), len(states)), -np.inf)
        for t in range(len(obs_seq) - 1):
            denom = logsumexp(alpha[t, :, None] + trans_prob + emit_prob[:, obs_seq[t+1]] + beta[t+1, :], axis=None)
            for i in range(len(states)):
                num = alpha[t, i] + trans_prob[i, :] + emit_prob[:, obs_seq[t+1]] + beta[t+1, :]
                xi[t, i, :] = num - denom

        gamma = (alpha + beta) - logsumexp(alpha + beta, axis=1, keepdims=True)

        new_trans_prob = logsumexp(xi, axis=0) - logsumexp(gamma[:-1], axis=0)
        new_emit_prob = np.full_like(emit_prob, -np.inf)
        for i in range(len(states)):
            denom = logsumexp(gamma[:, i])
            for o in range(emit_prob.shape[1]):
                mask = obs_seq == o
                if np.any(mask):
                    new_emit_prob[i, o] = logsumexp(gamma[mask, i]) - denom

        if np.allclose(np.exp(new_trans_prob), np.exp(prev_trans_prob), atol=convergence_threshold) and \
           np.allclose(np.exp(new_emit_prob), np.exp(prev_emit_prob), atol=convergence_threshold):
            break

        prev_trans_prob, prev_emit_prob = new_trans_prob, new_emit_prob
        trans_prob, emit_prob = new_trans_prob, new_emit_prob
        iteration += 1

        #if iteration == 3:
        #    break

    return trans_prob, emit_prob, iteration

# states = [0, 1]
# obs_seq = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
# obs_seq = np.array(obs_seq)

# start_prob = np.log([0.6, 0.4])
# trans_prob = np.log([[0.7, 0.3], [0.4, 0.6]])
# emit_prob = np.log([[0.9, 0.1], [0.2, 0.8]])

# trans_prob, emit_prob, iterations = forward_backward(obs_seq, states, start_prob, trans_prob, emit_prob)

# print("Learned Transition Probabilities:")
# print(np.exp(trans_prob))  
# print("Learned Emission Probabilities:")
# print(np.exp(emit_prob))  
# print("Iterations until convergence:")
# print(iterations)

states = np.array(['A', 'B', 'C'])
observations = np.array([0, 1, 2, 3])

start_prob = np.log(np.array([0.5, 0.3, 0.2]))

trans_prob = np.log(np.array([[0.6, 0.2, 0.2],
                              [0.3, 0.4, 0.3],
                              [0.3, 0.3, 0.4]]))

emit_prob = np.log(np.array([[0.4, 0.3, 0.2, 0.1],
                             [0.2, 0.4, 0.3, 0.1],
                             [0.1, 0.1, 0.4, 0.4]]))

obs_seq = np.array([0, 1, 2, 2, 0, 1, 1, 2, 0])

new_trans_prob, new_emit_prob, iteration = forward_backward(obs_seq, states, start_prob, trans_prob, emit_prob)

print("Updated Transition Probabilities:\n", np.exp(new_trans_prob))
print("Updated Emission Probabilities:\n", np.exp(new_emit_prob))
print("Iterations:", iteration)