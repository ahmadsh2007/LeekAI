import time
import sys
import numpy as np

from .layers import softmax_loss

# --- MODIFIED: Added output_stream parameter ---
def train(model, X_train, y_train, epochs, learning_rate, batch_size=32, clip_grad=None, lr_decay=0.95, output_stream=None):
    """
    Trains the CNN model using mini-batch gradient descent.

    Args:
        model: The CNNModel instance.
        X_train: Training images (N, C, H, W).
        y_train: Training labels (N,).
        epochs (int): Number of epochs.
        learning_rate (float): Initial learning rate.
        batch_size (int): Size of mini-batches.
        clip_grad (float, optional): Value to clip gradients. Defaults to None.
        lr_decay (float): Multiplicative factor for LR decay per epoch.
        output_stream (object, optional): A file-like object (like sys.stdout or a queue)
                                          to write progress messages to. Defaults to sys.stdout.
    """
    # --- Use provided stream or default to stdout ---
    out = output_stream if output_stream else sys.stdout

    # --- Write initial messages to the stream ---
    out.write(f"Starting training...\n")
    out.write(f"  Epochs: {epochs}\n")
    out.write(f"  Learning Rate: {learning_rate} (decay={lr_decay})\n")
    out.write(f"  Batch Size: {batch_size}\n")
    # Use getattr for safety, model might be partially initialized on error
    model_name = getattr(model, 'architecture_name', 'Unknown')
    out.write(f"  Model: {model_name}\n")
    out.flush() # Ensure initial messages appear

    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)

    # --- Variables for average batch time calculation ---
    batch_times = []
    avg_batch_time = 0.1 # Initial estimate

    current_lr = learning_rate # Use a separate variable for decaying LR

    for epoch in range(epochs):
        t_epoch_start = time.time()

        # Shuffle data indices at the start of each epoch
        np.random.shuffle(indices)
        X_train_shuf = X_train[indices]
        y_train_shuf = y_train[indices]

        total_loss = 0.0
        correct = 0

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            t_batch_start = time.time()

            # Get batch indices
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            X_batch = X_train_shuf[start:end]
            y_batch = y_train_shuf[start:end]

            # --- Forward Pass ---
            Z_out, caches = model.forward(X_batch)

            # --- Compute Loss and Gradient ---
            loss, dZ, probs = softmax_loss(Z_out, y_batch)
            total_loss += loss * X_batch.shape[0]

            # --- Backward Pass ---
            model.backward(dZ, caches)

            # --- Update Weights ---
            model.update(current_lr, clip=clip_grad) # Use current_lr

            # --- Calculate accuracy for batch ---
            preds = np.argmax(probs, axis=1)
            correct += np.sum(preds == y_batch)

            # --- Calculate timing ---
            batch_time = time.time() - t_batch_start
            batch_times.append(batch_time)
            if len(batch_times) > 20: # Keep last 20 timings for smoother avg
                 batch_times.pop(0)
            avg_batch_time = sum(batch_times) / len(batch_times)
            eta_s = (num_batches - (i + 1)) * avg_batch_time


            # --- Print Progress Bar to the stream ---
            progress = (i + 1) / num_batches
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "-" * (bar_len - filled)

            progress_str = (
                f"Epoch {epoch+1}/{epochs} |{bar}| {progress:.1%} "
                f"| Batch Time: {batch_time:.2f}s | Avg: {avg_batch_time:.2f}s | ETA: {int(eta_s)}s "
            )
            padding = " " * (85 - len(progress_str)) # Adjust padding slightly
            out.write("\r" + progress_str + padding)
            out.flush()


        # --- End of epoch ---
        epoch_time = time.time() - t_epoch_start
        avg_loss = total_loss / num_samples
        acc = correct / num_samples

        # --- Print epoch summary to the stream (overwrites progress bar) ---
        epoch_summary = (
            f"Epoch {epoch+1} complete. "
            f"Time: {epoch_time:.2f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Accuracy: {acc:.2%}"
        )
        padding = " " * (85 - len(epoch_summary))
        out.write("\r" + epoch_summary + padding + "\n") # Add newline here
        out.flush()


        # Decay learning rate for the next epoch
        current_lr *= lr_decay

    out.write("Training complete.\n") # Final message
    out.flush()