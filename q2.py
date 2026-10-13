import numpy as np

def softmax(z):
    """Numerically stable softmax along the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    z = np.array([1000.0, 1000.0, 1001.0])
    print(softmax(z))