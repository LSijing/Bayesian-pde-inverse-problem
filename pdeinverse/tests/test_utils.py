import unittest
from pdeinverse import utils


class MyTestCase(unittest.TestCase):
    def test_pca_1(self):
        import numpy as np
        x = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [1, 0, -1, 0, 1]]).transpose()
        n, m = x.shape
        num_basis = 2
        v, w = utils.compute_PCA(x, k=num_basis, mean=True)
        norms = np.diag(v.transpose() @ v)
        self.assertEqual(v.shape, (n, num_basis + 1))
        self.assertEqual(w.shape, (m,))
        # print(norms)
        self.assertEqual(np.allclose(norms[1:], np.ones(num_basis)), True)
        y = x - np.tile(v[:, 0].reshape((-1, 1)), (1, m))
        # print(y)
        # wp, vp = np.linalg.eigh(x @ x.transpose())
        # print(y.shape, (y @ y.T @ v[:, 1:2] / v[:, 1:2]).shape)
        # print(np.isclose(y, v[:, 1:2] @ v[:, 1:2].T @ y + v[:, 2:3] @ v[:, 2:3].T @ y))
        self.assertEqual(np.allclose(y, v[:, 1:2] @ v[:, 1:2].T @ y + v[:, 2:3] @ v[:, 2:3].T @ y), True)

        n, m = 100, 10
        x = np.random.randn(n, m)
        num_basis = 9
        v, w = utils.compute_PCA(x, k=num_basis, mean=True)
        norms = np.diag(v.transpose() @ v)
        self.assertEqual(v.shape, (n, num_basis + 1))
        self.assertEqual(w.shape, (m,))
        # print(norms)
        self.assertEqual(np.allclose(norms[1:], np.ones(num_basis)), True)
        y = x - np.tile(v[:, 0].reshape((-1, 1)), (1, m))
        # print(y)
        # wp, vp = np.linalg.eigh(x @ x.transpose())
        # print(y.shape, (y @ y.T @ v[:, 1:2] / v[:, 1:2]).shape)
        # print(np.isclose(y, v[:, 1:2] @ v[:, 1:2].T @ y + v[:, 2:3] @ v[:, 2:3].T @ y))
        # self.assertEqual(np.allclose(y, v[:, 1:2] @ v[:, 1:2].T @ y + v[:, 2:3] @ v[:, 2:3].T @ y), True)


if __name__ == '__main__':
    unittest.main()
