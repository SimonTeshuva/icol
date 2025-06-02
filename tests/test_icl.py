import numpy as np
import pytest

from icol.icol import SIS

def get_corrs(X, y):
    """
    Return pearson between X and y
    """
    return np.corrcoef(X, y)#[1, :-1]

def test_corrs():
    X = np.array([
        [1, 0.5],
        [2, 1],
        [3, 2]
    ])
    y = np.array([1, 2, 3])
    truth = get_corrs(X.T, y)
    print(truth)
    truth.sort()
    sis = SIS(X.shape[1])
    c, idx = sis(X=X, res=y, pool=[])
    c.sort()

    assert np.allclose(truth, c, atol=1e-6)

# def test_SIS():
#     random_state = 0
#     n = 20
#     p = 10
#     s = 3
#     np.random.seed(random_state)
#     X = np.array([
#         [0, -1, -2, 0, 0, 0, 0, 0, 0, 0],
#         [1, -2, -0.5, 0, 0, 0, 0, 0, 0, 0],
#         [2, -3, 1, 0, 0, 0, 0, 0, 0, 0],
#         [3, -4, 2.5, 0, 0, 0, 0, 0, 0, 0],
#         [4, -5, 4, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     y = X[:, 0] + 2*X[:, 1] + 3*X[:, 2]
    # print(y)
    # truth = get_corrs(X, y)
    # print(truth)
    # truth = np.abs(truth)
    # truth_idx = np.argsort(truth)
    # truth_idx.sort()
    # best_idx = truth_idx[:s]

    # sis = SIS(n_sis=s)
    # c, idx = sis(X=X, res=y, pool=[])
    # idx.sort()

    # assert np.allclose(best_idx, idx, atol=1e-6)
# test_SIS()
# test_corrs()

if __name__ == "__main__":
    pass
    # import pytest
    # pytest.main([__file__])