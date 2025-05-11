import numpy as np

def project_onto_hyperplane(w, w1):
    """Project w onto the hyperplane orthogonal to w1."""
    return w - (np.dot(w, w1)) * w1

def project_onto_nonnegative(w):
    """Set negative entries to zero."""
    return np.maximum(w, 0)

def project_onto_sphere(w):
    """Project to unit norm (if non-zero)."""
    norm = np.linalg.norm(w)
    if norm > 0:
        return w / norm
    else:
        return np.ones_like(w) / np.sqrt(len(w))

def nonnegative_second_pc(S, w1, n_iter=1000, lr=1e-2):
    d = S.shape[0]
    w = np.random.rand(d)
    w = project_onto_hyperplane(w, w1)
    w = project_onto_nonnegative(w)
    w = project_onto_sphere(w)

    for _ in range(n_iter):
        grad = 2 * S @ w
        w = w + lr * grad

        # Sequential projections
        w = project_onto_hyperplane(w, w1)
        w = project_onto_nonnegative(w)
        w = project_onto_sphere(w)

    return w
