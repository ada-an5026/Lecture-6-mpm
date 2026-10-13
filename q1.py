import numpy as np

def bigram_counts(ids, V):
    """Count adjacent pairs. N[i, j] = number of times j followed i."""
    N = np.zeros((V, V), dtype=np.int64)
    np.add.at(N, (ids[:-1], ids[1:]), 1)
    return N


text = "abracadabra"
chars = sorted(set(text))      
print(chars)              # ['a', 'b', 'c', 'd', 'r']
stoi = {c: i for i, c in enumerate(chars)}
print(stoi)
ids = np.array([stoi[c] for c in text])
print(ids)

N = bigram_counts(ids, len(chars))
print(N)
print(N.sum(axis=1, keepdims=True))
P = N / N.sum(axis=1, keepdims=True)
print(P)