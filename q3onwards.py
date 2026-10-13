import numpy as np

## Q6 note --> Smoothing adds +1 to all the unique tokens/characters in the bigram
## Q8 note --> The pairs are merged from most frequent token to least. Hence byte 256 will have the most freq token and then the next most freq

def softmax(z):
    '''Numerically stable softmax along the last axis.'''
    z = z - z.max(axis=-1, keepdims=True)     # subtract the largest logit
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def sample_next(probs, rng, temperature=1.0, top_k=None):
    '''Draw one token id from `probs`, with temperature and optional top-k.'''
    logits = np.log(probs + 1e-12) / temperature

    if top_k is not None:
        threshold = np.sort(logits)[-top_k]
        logits = np.where(logits < threshold, -np.inf, logits)
    
    print(softmax(logits))

    return rng.choice(len(probs), p=softmax(logits))

P = np.array([[0.25,  0.5,  0.25 ],     # P[i, j] = P(next = j | current = i)
              [0.5,   0.25, 0.25 ],
              [0.125, 0.75, 0.125]])
ids = np.array([0, 1, 2, 0])

def nll(P, ids):
    """Mean negative log-likelihood per character of `ids` under bigram model P."""
    return -np.log(P[ids[:-1], ids[1:]]).mean()

print(np.exp(nll(P, ids)))  # should be close to 1.0