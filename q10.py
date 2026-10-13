import numpy as np

def softmax(z):
    '''Numerically stable softmax along the last axis.'''
    z = z - z.max(axis=-1, keepdims=True)     # subtract the largest logit
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=True):
    """Single-head self-attention."""
    Q, K, Vv = X @ Wq, X @ Wk, X @ Wv

    scores = Q @ K.T / np.sqrt(Wk.shape[1])

    if causal:
        T = X.shape[0]
        scores = np.where(np.tril(np.ones((T, T))) == 0, -np.inf, scores)

    A = softmax(scores)
    return A @ Vv, A

T, d, d_k = 5, 6, 4
X = np.random.default_rng(0).normal(size=(T, d))
out, A = attention(X, np.zeros((d, d_k)), np.zeros((d, d_k)), np.eye(d))

print(A[2])