import numpy as np
import re
from scipy.special import logsumexp

def custom_split(sentence):
    tokens = re.findall(r'\w+|[,.!?;"]', sentence)
    return tokens

class HMMUnsupervised:
    def __init__(self, num_states) -> None:
        self.num_states = num_states
        #self.num_obs = num_obs

        self.start_prob = np.log(np.random.dirichlet(np.ones(num_states)))
        self.trans_prob = np.log(np.random.dirichlet(np.ones(num_states), size=num_states))
        self.emit_prob = None

        self.vocab = {}

    def reader(self, data):
        processed_data = []
        for line in data:
            line = re.sub(r"^\S+::\d+\s+", "", line)
            words = custom_split(line)
            word_indices = []
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                word_indices.append(self.vocab[word])
            processed_data.append(np.array(word_indices, dtype=np.int32))

        num_obs = len(self.vocab)
        self.emit_prob = np.log(np.random.dirichlet(np.ones(num_obs), size=self.num_states))
        
        return processed_data
    
    def forward(self, obs_seq):
        alpha = np.full((len(obs_seq), self.num_states), -np.inf)
        alpha[0, :] = self.start_prob + self.emit_prob[:, obs_seq[0]]
        for t in range(1, len(obs_seq)):
            for j in range(self.num_states):
                alpha[t, j] = logsumexp(alpha[t - 1] + self.trans_prob[:, j]) \
                    + self.emit_prob[j, obs_seq[t]]

        return alpha
    
    def backward(self, obs_seq):
        beta = np.full((len(obs_seq), self.num_states), -np.inf)
        beta[-1, :] = 0
        for t in range(len(obs_seq) - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = logsumexp(beta[t + 1] + self.trans_prob[i, :] \
                                       + self.emit_prob[:, obs_seq[t + 1]])

        return beta
    
    def forward_backward(self, obs_seq, convergence_threshold=1e-6):
        prev_trans_prob = np.copy(self.trans_prob)
        prev_emit_prob = np.copy(self.emit_prob)
        iteration = 0

        while True:
            alpha = self.forward(obs_seq)
            beta = self.backward(obs_seq)

            xi = np.full((len(obs_seq) - 1, self.num_states, self.num_states), \
                          -np.inf)
            for t in range(len(obs_seq) - 1):
                denom = logsumexp(alpha[t, :, None] + self.trans_prob \
                                  + self.emit_prob[:, obs_seq[t+1]] + beta[t+1, :], axis=None)
                for i in range(self.num_states):
                    num = alpha[t, i] + self.trans_prob[i, :] + self.emit_prob[:, obs_seq[t+1]] + \
                          beta[t+1, :]
                    xi[t, i, :] = num - denom

            gamma = (alpha + beta) - logsumexp(alpha + beta, axis=1, keepdims=True)
            new_trans_prob = logsumexp(xi, axis=0) - logsumexp(gamma[:-1], axis=0)
            new_emit_prob = np.full_like(self.emit_prob, -1e10)
            for i in range(self.num_states):
                denom = logsumexp(gamma[:, i])
                for o in range(self.emit_prob.shape[1]):
                    mask = obs_seq == o
                    if np.any(mask):
                        new_emit_prob[i, o] = logsumexp(gamma[mask, i]) - denom

            if np.allclose(np.exp(new_trans_prob), np.exp(prev_trans_prob), atol=convergence_threshold) and \
                np.allclose(np.exp(new_emit_prob), np.exp(prev_emit_prob), atol=convergence_threshold):
                break

            prev_trans_prob, prev_emit_prob = new_trans_prob, new_emit_prob
            self.trans_prob, self.emit_prob = new_trans_prob, new_emit_prob
            iteration += 1

            # if iteration ==3:
            #     break
            
        return iteration
    
    def learn_sentences(self, epoch, processed_data):
        for epoch in range(epoch):
            for sequence in processed_data:
                print(sequence)
                self.forward_backward(sequence)
            print(f"Epoch {epoch+1} completed.")

data = [
    "word1::123 Hello world is a test",
    "word2::123 This is a test",
    "word3::123 This is Another example"
]

num_states = 5
hmm = HMMUnsupervised(num_states)


processed_data = hmm.reader(data)


epochs = 1
hmm.learn_sentences(epochs, processed_data)