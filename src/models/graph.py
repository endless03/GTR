import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from tqdm import tqdm


def build_relation_graph_raw_sequence(data, threshold=0.3, topk=None,
                                      alpha=0.05, n_perm=100, n_jobs=-1):
    """
    Build relation graph from raw sequence

    Parameters:
    -----------
    data : np.ndarray
        Input data with shape [length, D]
    threshold : float
        Mutual information threshold
    topk : int
        Maximum connections per node
    alpha : float
        Significance level
    n_perm : int
        Number of permutation tests
    n_jobs : int
        Number of parallel jobs

    Returns:
    --------
    edge_index : torch.Tensor
        Edge indices with shape [2, E]
    edge_attr : torch.Tensor
        Edge attributes with shape [E, 1]
    """
    assert data.ndim == 2, "Data must be 2D array with shape [length, D]"
    length, D = data.shape

    # Compute mutual information matrix
    mi_matrix = np.zeros((D, D))

    def calc_mi(i, j):
        """Calculate mutual information between two dimensions"""
        if i == j:
            return 0.0
        X = data[:, i].reshape(-1, 1)
        y = data[:, j]
        return mutual_info_regression(X, y, n_neighbors=3)[0]

    # Parallel computation of mutual information
    for i in tqdm(range(D), desc='Calculating MI matrix'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(calc_mi)(i, j) for j in range(i + 1, D)
        )
        for idx, j in enumerate(range(i + 1, D)):
            mi = results[idx]
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # Normalize mutual information
    mi_max = np.max(mi_matrix)
    if mi_max == 0:
        mi_norm = np.zeros_like(mi_matrix)
    else:
        mi_norm = mi_matrix / mi_max

    # Build edges
    edges = []
    for i in range(D):
        candidates = []
        for j in range(D):
            if i == j:
                continue
            if mi_norm[i, j] > threshold:
                candidates.append((j, mi_norm[i, j]))

        # Apply topk constraint
        if topk is not None and len(candidates) > topk:
            candidates = sorted(candidates, key=lambda x: -x[1])[:topk]

        edges += [(i, j, score) for (j, score) in candidates]

    # Convert to PyG format
    if len(edges) == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1)

    edge_index = torch.tensor([(i, j) for i, j, _ in edges],
                              dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([score for _, _, score in edges],
                             dtype=torch.float32).unsqueeze(1)

    return edge_index, edge_attr