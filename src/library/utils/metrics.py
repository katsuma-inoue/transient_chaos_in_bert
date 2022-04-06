"""Implement functions for some metrics.
"""
import numpy as np
import tensorflow as tf


def effective_dimension(X: np.ndarray, gpu_idx: int = -1):
    """Calculate effective dimension of the current state.
    Because of the large size of array, this function will
    consume memory dramatically.

    Parameters
    ----------
    X : np.ndarray
        (num_samples, hidden_dim*num_token). The hidden state of specified timestep.
    gpu_idx : int, optional
        If bigger than -1, calculation will be executed with
        specified gpu., by default -1

    Returns
    -------
    float
        Calculated effective dimension.
    """
    if gpu_idx > -1:
        with tf.device(f'/device:GPU:{gpu_idx}'):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            mean = tf.reduce_mean(X, axis=0, keepdims=True)
            X -= mean
            X_t = tf.transpose(X)
            eigs = tf.linalg.eigh(tf.tensordot(X_t, X, axes=[[1], [0]]))[0]
            es = tf.math.abs(eigs)
            es *= 1 / tf.reduce_sum(es)
            result = 1 / tf.reduce_sum(es**2)
        return result.numpy()
    else:
        eigs = np.linalg.eig(X.T.dot(X))[0]
        es = abs(eigs)
        es *= 1 / sum(es)
        return 1 / (sum(es**2))
