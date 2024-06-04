import time
from tqdm import tqdm


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

