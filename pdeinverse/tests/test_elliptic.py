import unittest
from pdeinverse import elliptic
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_uniform_triangulation(self):
        size = np.random.randint(64) + 1
        points, tris, edges, x_mesh, y_mesh = elliptic.compute_uniform_triangulation_vmatlab(n=size)
        # test shapes
        self.assertEqual(points.shape, ((size+1) ** 2, 2))
        self.assertEqual(tris.shape, (size ** 2 * 2, 3))
        self.assertEqual(edges.shape, (size * 4, 2))
        self.assertEqual(x_mesh.shape, (size + 1, size + 1))
        self.assertEqual(y_mesh.shape, (size + 1, size + 1))
        # test output
        pos = np.random.randint((size + 1) ** 2)
        self.assertEqual(np.allclose(points[pos], [(pos % (size + 1)) / size, (pos // (size + 1)) / size]), True)
        i, j = np.random.randint(size), np.random.randint(size)
        self.assertEqual(np.allclose(points[i+j*(size+1)], [y_mesh[i,j], x_mesh[i,j]]), True)
        self.assertEqual(True, True)

    def test_compute_mass_matrix(self):
        size = 15
        points, tris, edges, x_mesh, y_mesh = elliptic.compute_uniform_triangulation_vmatlab(n=size)
        mass_matrix = elliptic.compute_mass_matrix(tris, points)
        self.assertEqual(mass_matrix.nnz, (size ** 2 * 2) * 9)
        self.assertEqual(np.isclose(np.max(mass_matrix), 1/(2 * size**2)), True)
        self.assertEqual(mass_matrix.shape, ((size+1)**2, (size+1)**2))
        self.assertEqual(np.isclose(np.sum(mass_matrix.data), 1), True)

    def test_get_int_bd_idx(self):
        size = 4
        points, tris, edges, x_mesh, y_mesh = elliptic.compute_uniform_triangulation_vmatlab(n=size)
        in_idx, bd_idx = elliptic.get_int_bd_idx(tris, edges)
        self.assertEqual(in_idx.shape, ((size - 1) ** 2,))
        self.assertEqual(bd_idx.shape, (4 * size,))

    def test_compute_load_vector(self):
        size = 10
        points, tris, _, _, _ = elliptic.compute_uniform_triangulation_vmatlab(n=size)
        def foo(x, y):
            return np.sin(x) + np.exp(x) + np.cos(2*np.pi*y)
        b1 = elliptic.compute_load_vector(tris, points, foo, quad='midpoint')
        b2 = elliptic.compute_load_vector(tris, points, foo, quad='corner')
        b3 = elliptic.compute_load_vector(tris, points, foo, quad='center')
        self.assertEqual(b1.shape, ((size + 1) ** 2, 1))
        self.assertEqual(b2.shape, ((size + 1) ** 2, 1))
        self.assertEqual(b3.shape, ((size + 1) ** 2, 1))
        norms = np.concatenate((b1.toarray().T @ b1.toarray(), b3.toarray().T @ b3.toarray(),b3.toarray().T @ b3.toarray()))
        self.assertEqual(np.all((b1.toarray() - b2.toarray()).T @ (b1.toarray() - b2.toarray())/ np.mean(norms) < 1/(size ** 2)), True)
        self.assertEqual(np.all((b1.toarray() - b3.toarray()).T @ (b1.toarray() - b3.toarray())/ np.mean(norms) < 1/(size ** 2)), True)

    def test_compute_stiffness_matrix(self):
        size = 5
        points, tris, _, _, _ = elliptic.compute_uniform_triangulation_vmatlab(n=size)
        coef = np.ones(size ** 2 * 2)
        stiff_mat = elliptic.compute_stiffness_matrix(tris, points, coef)
        self.assertEqual(stiff_mat.nnz, (size ** 2 * 2)* 9)
        self.assertEqual(stiff_mat.shape, ((size + 1) ** 2, (size + 1) ** 2))

    def test_compute_pde_dictionary(self):
        n = 3
        pde = elliptic.compute_pde_dictionary(n)
        points, tris, _, _, _ = elliptic.compute_uniform_triangulation_vmatlab(n)
        x1_elem = points[tris[:, 0], 0:1]
        x2_elem = points[tris[:, 1], 0:1]
        x3_elem = points[tris[:, 2], 0:1]
        y1_elem = points[tris[:, 0], -1:]
        y2_elem = points[tris[:, 1], -1:]
        y3_elem = points[tris[:, 2], -1:]
        xc_elem = (x1_elem + x2_elem + x3_elem) / 3
        yc_elem = (y1_elem + y2_elem + y3_elem) / 3
        self.assertEqual(np.allclose(pde['points'], points), True)
        self.assertEqual(np.allclose(pde['tris'], tris), True)
        self.assertEqual(np.count_nonzero(pde['fixed_node'] <= n), n+1)
        self.assertEqual(np.count_nonzero(pde['fixed_node'] >= n * (n+1)), n + 1)
        self.assertEqual(pde['fixed_node'].size, 2*n+2)
        self.assertEqual(pde['free_node'].size + pde['fixed_node'].size, (n+1) ** 2)
        self.assertEqual(pde['center'].shape, (n ** 2 * 2, 2))
        self.assertEqual(np.allclose(np.concatenate((xc_elem, yc_elem), axis=1), pde['center']), True)
        self.assertEqual(pde['g_D'].shape, (2 * n + 2, ))

if __name__ == '__main__':
    unittest.main()
