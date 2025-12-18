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


def train_transformer(model, x_train, y_train, learning_rate=0.01, nepoch=100, evaluation_loss_after=5, batch_size=32,
                      print_every=1, clip_grad_norm=None, idx_to_word=None, sample_prompts=None, topk=5):
    """
    Vectorized mini-batch training loop for the numpy/cupy Transformer.
    - Uses the model's backend (`model.np`) everywhere to avoid CPU/GPU mixing.
    - Dynamically trims each batch to its max non-pad length to cut S^2 attention cost.
    - Computes loss only on each sequence's last real token (next-word prediction).
    - Optional gradient clipping and simple progress samples.
    """

    npb = model.np
    losses = []
    seen_examples = 0
    start_time = time.time()

    # Ensure backend arrays
    x_train = npb.array(x_train)
    y_train = npb.array(y_train)

    N = len(y_train)

    def compute_eval_loss(x_data, y_data):
        total_loss = 0.0
        total_count = 0
        model.eval()
        for i in range(0, len(y_data), batch_size):
            xb = x_data[i: i + batch_size]
            yb = y_data[i: i + batch_size]

            # per-batch dynamic trim by non-pad length
            lens = npb.sum(xb != 0, axis=1).astype(npb.int32)
            if lens.size == 0:
                continue
            s_max = int(npb.max(lens))
            xb = xb[:, :s_max]

            H, _ = model.forward(xb, return_hidden_only=True)  # (B, s_max, E)
            idx = npb.arange(len(yb))
            t = lens - 1
            H_last = H[idx, t, :]  # (B,E)
            logits_last = H_last @ model.W_vocab + model.b_vocab  # (B,V)
            loss, _ = model.calculate_loss_dlogits(logits_last, yb)
            total_loss += float(loss) * len(yb)
            total_count += len(yb)
        model.train()
        return total_loss / max(1, total_count)

    # epoch loop
    for epoch in range(nepoch):
        # Evaluate
        if epoch % evaluation_loss_after == 0:
            eval_loss = compute_eval_loss(x_train, y_train)
            losses.append((seen_examples, eval_loss))
            elapsed = time.time() - start_time
            ppl = np.exp(eval_loss)
            print(f"Eval loss={eval_loss:.6f} | ppl={ppl:.2f} | seen={seen_examples} | epoch={epoch} | {elapsed:.1f}s")
            if len(losses) > 1 and eval_loss > losses[-2][1]:
                learning_rate *= 0.5
                print("Learning rate decreased to:", learning_rate)
            # Optionally release CuPy memory pool after large eval batch
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        # Shuffle each epoch
        perm = npb.random.permutation(N)
        x_train = x_train[perm]
        y_train = y_train[perm]

        token_done = 0
        with tqdm(total=N, desc=f"Epoch {epoch + 1}/{nepoch}") as pbar:
            for i in range(0, N, batch_size):
                xb = x_train[i: i + batch_size]
                yb = y_train[i: i + batch_size]

                lens = npb.sum(xb != 0, axis=1).astype(npb.int32)
                s_max = int(npb.max(lens))
                xb = xb[:, :s_max]

                H, cache = model.forward(xb, return_hidden_only=True)  # (B, s_max, E)
                idx = npb.arange(len(yb))
                t = lens - 1
                H_last = H[idx, t, :]  # (B,E)
                logits_last = H_last @ model.W_vocab + model.b_vocab  # (B,V)

                loss, dlog_last = model.calculate_loss_dlogits(logits_last, yb)
                # dlog_last: (B,1,V) because calculate_loss_dlogits normalizes shapes
                grads = model.backward_last_token(cache, t, dlog_last[:, 0, :])

                if clip_grad_norm is not None:
                    # Flatten all gradient arrays into a list for clipping
                    grad_list = []
                    for g in grads.values():
                        grad_list.append(g)
                    # clip_grads operates in-place
                    from utils.function import clip_grads
                    clip_grads(grad_list, clip_grad_norm, lib=npb)

                model.step(grads, learning_rate)

                seen_examples += len(yb)
                token_done += int(npb.sum(lens))
                pbar.update(len(yb))
                if print_every and ((i // batch_size) % print_every == 0):
                    pbar.set_postfix({
                        'loss': float(loss),
                        'tokens': token_done
                    })

        # Simple qualitative sampling each epoch
        if idx_to_word is not None and sample_prompts:
            try:
                print("\nSamples:")
                for prompt in sample_prompts:
                    seq = npb.array(prompt, dtype=npb.int32)[None, :]
                    H, _ = model.forward(seq, return_hidden_only=True)
                    next_logits = H[0, -1] @ model.W_vocab + model.b_vocab
                    probs = npb.exp(next_logits - next_logits.max())
                    probs /= probs.sum()
                    # Bring to CPU for argsort display if needed
                    probs_cpu = probs
                    if hasattr(npb, 'asnumpy'):
                        try:
                            probs_cpu = npb.asnumpy(probs)
                        except Exception:
                            pass
                    top_idx = np.array(probs_cpu).argsort()[-topk:][::-1]
                    words = [idx_to_word.get(int(i), '<UNK>') for i in top_idx]
                    print('  ->', ' '.join(words))
            except Exception as e:
                print('Sampling error:', e)

        # Release unused GPU memory chunks between epochs
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

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
