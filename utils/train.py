import time
from tqdm import tqdm
import numpy as np


def train_with_sgd(model, x_train, y_train, learning_rate=0.01, nepoch=100, evaluation_loss_after=5):
    losses = []
    num_examples = 0
    start_time = time.time()

    for epoch in range(nepoch):
        if epoch % evaluation_loss_after == 0:
            loss = model.calculate_loss(x_train, y_train)
            losses.append((num_examples, loss))
            time_elapsed = time.time() - start_time
            print(
                f"Loss after num_examples={num_examples}, epoch={epoch}: {loss:.6f} - Time elapsed: {time_elapsed:.2f} sec")
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
                print("Learning rate decreased to:", learning_rate)

        for i in tqdm(range(len(y_train)), desc=f"Epoch {epoch + 1}/{nepoch} Progress"):
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_examples += 1

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} sec")


def train_transformer(model, x_train, y_train, learning_rate=0.01, nepoch=100, evaluation_loss_after=5, batch_size=32):
    np_backend = model.np
    losses = []
    num_examples = 0
    start_time = time.time()

    x_train = np_backend.array(x_train)
    y_train = np_backend.array(y_train)

    def compute_loss_for_batches(x_data, y_data):
        total_loss = 0.0
        count = 0
        for i in range(0, len(y_data), batch_size):
            x_batch = x_data[i: i + batch_size]
            y_batch = y_data[i: i + batch_size]
            logits, _ = model.forward(x_batch)
            logits_last = logits[:, -1:, :]
            loss, _ = model.calculate_loss_dlogits(logits_last, y_batch[:, None])
            total_loss += loss * len(y_batch)
            count += len(y_batch)
        return total_loss / count


    for epoch in range(nepoch):
        if epoch % evaluation_loss_after == 0:
            eval_loss = compute_loss_for_batches(x_train, y_train)
            losses.append((num_examples, eval_loss))
            time_elapsed = time.time() - start_time
            print(
                f"Loss after num_examples={num_examples}, epoch={epoch}: {eval_loss:.6f} - Time elapsed: {time_elapsed:.2f} sec")
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
                print("Learning rate decreased to:", learning_rate)

        for i in tqdm(range(len(y_train)), desc=f"Epoch {epoch + 1}/{nepoch}"):
            x_sample, y_sample = x_train[i], y_train[i]
            x_batch, y_batch = np.expand_dims(x_sample, axis=0), np.expand_dims(y_sample, axis=0)

            logits, cache = model.forward(x_batch)

            logits_last = logits[:, -1:, :]
            loss, dlogits_last = model.calculate_loss_dlogits(logits_last, y_batch[:, None])

            dlogits = np.zeros_like(logits)
            dlogits[:, -1, :] = dlogits_last[:, 0, :]

            grads = model.backward(cache, dlogits)
            model.sgd_step(grads, learning_rate)

            num_examples += 1

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} sec")


def train_skipgram(model, indexed_corpus, vocab_size, window_size=2, epochs=10, learning_rate=0.01,
                   evaluation_interval=5):
    losses = []
    num_examples = 0
    start_time = time.time()

    for epoch in range(epochs):
        if (epoch + 1) % evaluation_interval == 0:
            loss = model.calculate_loss(indexed_corpus, vocab_size, window_size)
            losses.append((num_examples, loss))
            time_elapsed = time.time() - start_time
            print(
                f"\nLoss after {num_examples} examples, epoch {epoch + 1}: {loss:.6f} - Time elapsed: {time_elapsed:.2f} sec")

            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
                print("Learning rate decreased to:", learning_rate)

        for indexed_sentence in tqdm(indexed_corpus, desc=f'Epoch {epoch + 1}/{epochs}', leave=True):
            sentence_length = len(indexed_sentence)

            for i, target_index in enumerate(indexed_sentence):
                target = model.np.zeros(vocab_size)
                target[target_index] = 1
                context_indices = []

                for j in range(max(0, i - window_size), min(sentence_length, i + window_size + 1)):
                    if i != j:
                        context_index = indexed_sentence[j]
                        context_word = model.np.zeros(vocab_size)
                        context_word[context_index] = 1
                        context_indices.append(context_word)

                model.sgd_step(target, context_indices, learning_rate)
        num_examples += 1

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} sec")
