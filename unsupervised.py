import numpy as np
import re
from scipy.special import logsumexp

def custom_split(sentence):
    tokens = re.findall(r'\w+|[,.!?;"]', sentence)
    return tokens

class HMMUnsupervised:
    def __init__(self, num_states) -> None:
        self.num_states = num_states

        self.start_prob = np.log(np.random.dirichlet(np.ones(num_states)))
        self.trans_prob = np.log(np.random.dirichlet(np.ones(num_states), size=num_states))
        self.emit_prob = None

        self.vocab = {}
        self.state_frequencies = np.zeros(num_states)

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
        alpha = np.full((len(obs_seq), self.num_states), -1e8)
        alpha[0, :] = self.start_prob + self.emit_prob[:, obs_seq[0]]
        for t in range(1, len(obs_seq)):
            for j in range(self.num_states):
                alpha[t, j] = logsumexp(alpha[t - 1] + self.trans_prob[:, j]) \
                    + self.emit_prob[j, obs_seq[t]]

        return alpha
    
    def backward(self, obs_seq):
        beta = np.full((len(obs_seq), self.num_states), -1e8)
        beta[-1, :] = 0
        for t in range(len(obs_seq) - 2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = logsumexp(beta[t + 1] + self.trans_prob[i, :] \
                                       + self.emit_prob[:, obs_seq[t + 1]])

        return beta
    
    def forward_backward(self, obs_seq, convergence_threshold=1e9):
        prev_trans_prob = np.copy(self.trans_prob)
        prev_emit_prob = np.copy(self.emit_prob)
        iteration = 0

        while True:
            alpha = self.forward(obs_seq)
            beta = self.backward(obs_seq)
            print(self.trans_prob)
            print(self.emit_prob)

            xi = np.full((len(obs_seq) - 1, self.num_states, self.num_states), \
                          -1e8)
            for t in range(len(obs_seq) - 1):
                denom = logsumexp(alpha[t, :, None] + self.trans_prob \
                                  + self.emit_prob[:, obs_seq[t+1]] + beta[t+1, :], axis=None)
                for i in range(self.num_states):
                    num = alpha[t, i] + self.trans_prob[i, :] + self.emit_prob[:, obs_seq[t+1]] + \
                          beta[t+1, :]
                    xi[t, i, :] = num - denom

            gamma = (alpha + beta) - logsumexp(alpha + beta, axis=1, keepdims=True)

            # hidden state calculation logic, don't trust this number
            most_probable_states = np.argmax(gamma, axis=1)
            for state in most_probable_states:
                self.state_frequencies[state] += 1

            new_trans_prob = logsumexp(xi, axis=0) - logsumexp(gamma[:-1], axis=0)
            new_emit_prob = np.full_like(self.emit_prob, -1e8)
            for i in range(self.num_states):
                denom = logsumexp(gamma[:, i])
                for o in range(self.emit_prob.shape[1]):
                    mask = obs_seq == o
                    if np.any(mask):
                        new_emit_prob[i, o] = logsumexp(gamma[mask, i]) - denom

            if np.allclose(new_trans_prob, prev_trans_prob, atol=np.log(convergence_threshold)) and \
                np.allclose(new_emit_prob, prev_emit_prob, atol=np.log(convergence_threshold)):
                break


            prev_trans_prob, prev_emit_prob = new_trans_prob, new_emit_prob
            self.trans_prob, self.emit_prob = new_trans_prob, new_emit_prob
            iteration += 1

        return iteration
    
    def learn_sentences(self, epoch, processed_data):
        for epoch in range(epoch):
            for sequence in processed_data:
                print(sequence)
                self.forward_backward(sequence)
            print(f"Epoch {epoch+1} completed.")

    def save_results(self, filename='model_results.txt'):
        with open(filename, 'w') as file:
            state_freq_str = ', '.join(f"{i}: {int(freq)}" for i, freq in enumerate(self.state_frequencies))
            file.write(f"{{{state_freq_str}}}\n")
        
            obs_hidden_prob = {(obs, hidden): np.exp(prob) for hidden in range(self.num_states) for obs, prob in enumerate(self.emit_prob[hidden])}
            obs_hidden_str = ', '.join(f"({k[0]}, {k[1]}): {v:.4f}" for k, v in obs_hidden_prob.items())
            file.write(f"{{{obs_hidden_str}}}\n")
            
            hidden_hidden_prob = {(current, next): np.exp(prob) for current in range(self.num_states) for next, prob in enumerate(self.trans_prob[current])}
            hidden_hidden_str = ', '.join(f"({k[0]}, {k[1]}): {v:.4f}" for k, v in hidden_hidden_prob.items())
            file.write(f"{{{hidden_hidden_str}}}\n")

    def load_from_file(cls, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        state_freq_line = lines[0].strip('{}\n').split(': ')[1]
        state_frequencies = list(map(int, state_freq_line.split(', ')))

        num_states = len(state_frequencies)
        instance = cls(num_states)
        instance.state_frequencies = np.array(state_frequencies)

        obs_hidden_prob_line = lines[1].strip('{}()\n')
        instance.emit_prob = np.full((num_states, num_states), -np.inf) 
        for entry in obs_hidden_prob_line.split('), ('):
            pair, prob = entry.split(': ')
            obs, hidden = map(int, pair.split(', '))
            instance.emit_prob[hidden, obs] = float(prob)

        hidden_hidden_prob_line = lines[2].strip('{}()\n')
        for entry in hidden_hidden_prob_line.split('), ('):
            pair, prob = entry.split(': ')
            current, next = map(int, pair.split(', '))
            instance.trans_prob[current, next] = float(prob)

        return instance

def read_file_to_list(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        lines = [line.strip() for line in lines]
    return lines

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

hmm.save_results()