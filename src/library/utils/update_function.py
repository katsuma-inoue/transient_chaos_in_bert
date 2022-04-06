"""The update functions to emurate the network after synchronization
"""
import cupy as cp
import numpy as np


def gelu(x):
    """gelu activation.
    """
    cdf = 0.5 * (
        1.0 + cp.tanh(cp.sqrt(2.0 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))
    return x * cdf


def normalize(x, gamma, beta):
    """Normalize tensor x.
    https://github.com/kpe/params-flow/blob/master/params_flow/normalization.py
    Parameters :
    ------------------
    x : cp.ndarray. (B, H).
        B = Batch size
        H = Hidden size.
    """
    mean = cp.mean(x, axis=-1, keepdims=True)
    var = cp.var(x, axis=-1, keepdims=True)
    inv = gamma / cp.sqrt(var + 1e-12)
    x = x * inv + beta.astype(x.dtype) - (mean * inv).astype(x.dtype)
    return x


def update_state(x, weights, biases):
    """Update state vectors x.
    Parameters :
    ------------------
    x : ndarray. (B, H).
        B = Batch size
        H = Hidden size.
    """
    c = cp.dot(x, weights["V"]) + biases["V"]
    p = normalize(cp.dot(c, weights["d2"]) + biases["d2"] + x,
                  gamma=weights['n2'],
                  beta=biases['n2'])
    p_inter = gelu(cp.dot(p, weights["inter"]) + biases["inter"])
    x_ = normalize(cp.dot(p_inter, weights['d1']) + biases['d1'] + p,
                   gamma=weights['n1'],
                   beta=biases['n1'])
    return c, p, p_inter, x_
