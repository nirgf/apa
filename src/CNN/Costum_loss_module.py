import tensorflow as tf

def weighted_categorical_crossentropy(labels, logits, class_weights):
    """
    Computes the weighted categorical cross-entropy loss.

    Parameters:
        labels (Tensor): One-hot encoded true labels of shape (batch_size, height, width, num_classes).
        logits (Tensor): Logits predicted by the model of shape (batch_size, height, width, num_classes).
        class_weights (Tensor): Class weights of shape (num_classes,).

    Returns:
        Tensor: Scalar tensor representing the weighted loss.
    """
    # Ensure class_weights is broadcastable
    class_weights = tf.reshape(class_weights, [1, 1, 1, -1])  # Shape: (1, 1, 1, num_classes)

    # Compute weights for each pixel based on its true label
    weights = tf.reduce_sum(class_weights * labels, axis=-1)  # Shape: (batch_size, height, width)

    # Compute the unweighted softmax cross-entropy loss per pixel
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # Shape: (batch_size, height, width)

    # Apply weights to the losses
    weighted_losses = unweighted_losses * weights  # Shape: (batch_size, height, width)

    # Reduce the result to get the final mean loss
    loss = tf.reduce_mean(weighted_losses)  # Scalar loss

    return loss