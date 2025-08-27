import time
import sys
import random
import math


def calculate_loss(prediction, actual_label, eps=1e-12):
    p = prediction[actual_label]
    if p < eps:
        p = eps
    return -math.log(p)


def train(model, data, labels, epochs, learning_rate, clip_grad=None, lr_decay=0.95):
    print("Starting training with backpropagation...")
    total = len(data)

    for epoch in range(epochs):
        batch = list(zip(data, labels))
        random.shuffle(batch)
        data_shuf, labels_shuf = zip(*batch) if batch else ([], [])

        total_loss = 0.0
        correct = 0
        t0 = time.time()

        for i, (image, label) in enumerate(zip(data_shuf, labels_shuf)):
            if i % 20 == 0 or i == total - 1:
                progress = (i + 1) / max(1, total)
                elapsed = time.time() - t0
                eta = (elapsed / progress) - elapsed if progress > 0 else 0.0
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "█" * filled + "-" * (bar_len - filled)
                sys.stdout.write(
                    "\r"
                    + f"Epoch {epoch+1}/{epochs} |{bar}| {progress:.1%} | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
                )
                sys.stdout.flush()

            pred = model.forward(image)
            if pred and pred.index(max(pred)) == label:
                correct += 1

            loss = calculate_loss(pred, label)
            total_loss += loss

            model.backward(pred, label)
            model.update(learning_rate, clip=clip_grad)

        print()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(
            f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f} | Accuracy: {acc:.2%}"
        )

        learning_rate *= lr_decay

    print("Training complete.")
