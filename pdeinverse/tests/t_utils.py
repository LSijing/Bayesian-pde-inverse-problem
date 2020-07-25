
from pdeinverse import utils
import numpy as np

if __name__ == '__main__':
    n, m = 100, 10
    x = np.random.randn(n,m)
    v,w = utils.compute_PCA(x, k=5)
    print(v.shape, w)
    print(np.diag(v.transpose() @ v))
    y = x - np.tile(v[:,0].reshape((-1, 1)), (1, m))
    print((y @ y.transpose() @ v[:,1])/ v[:,1])

    # wp, vp = np.linalg.eigh(x @ x.transpose())
    # print( (x @ x.transpose() @ vp[:,-1]) / vp[:,-1])
