def train_with_sgd(model, x_train, y_train, learning_rate=0.01, nepoch=100, evaluation_loss_after=5):
    losses = []
    num_examples = 0
    for epoch in range(nepoch):
        if epoch % evaluation_loss_after == 0:
            loss = model.calculate_loss(x_train, y_train)
            losses.append((num_examples, loss))
            print(f"Loss after num_examples={num_examples}, epoch={epoch}: {loss:.6f}")
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
        for i in range(len(y_train)):
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_examples += 1
