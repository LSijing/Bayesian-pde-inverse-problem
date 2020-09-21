import unittest
from pdeinverse import hmc, elliptic
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_compute_inverse_dictionary_and_solve(self):
        n = 8
        var = 1.0
        corr_length = 0.2
        sigma_theta = 1.0
        noise_ob = 0.08
        kl_dim = 5
        pde = elliptic.compute_pde_dictionary(n)
        inv_pde = hmc.compute_inverse_pde_dictionary(pde, var=var, corr_length=corr_length, noise_ob=noise_ob, kl_ndim=kl_dim, sigma_theta=sigma_theta)
        print('\n', inv_pde.keys())
        # test inverse dictionary
        self.assertEqual(inv_pde['kl_ndim'], kl_dim)
        self.assertEqual(inv_pde['kl_weights'].shape, (kl_dim,))
        self.assertEqual(inv_pde['kl_modes'].shape, (2 * n ** 2, kl_dim))
        self.assertEqual(inv_pde['noise_ob'], noise_ob)
        self.assertEqual(inv_pde['sigma_theta'], sigma_theta)
        self.assertEqual(False, False)
        # test solve_from_normal_input
        normal = np.array([1, 0, 1, -1, 0], dtype=np.float)
        u,pu = hmc.solve_from_normal_input(inputs=normal, inv_pde=inv_pde, derivative=1)
        self.assertEqual(u.shape, ((n+1)**2, ))
        self.assertEqual(pu.shape, ((n+1) ** 2, kl_dim))
        # test observation operator
        observe_mat = hmc.get_observation_operator(n//2, inv_pde)
        self.assertEqual(observe_mat.shape, ((n//2 + 1) ** 2, (n + 1) ** 2))
        self.assertEqual(np.allclose(observe_mat @ u, u.reshape((n + 1, n + 1))[0::2, 0::2].flatten()), True)
        self.assertEqual(np.allclose(observe_mat @ pu[:, 2], pu[:, 2].reshape((n + 1, n + 1))[0::2, 0::2].flatten()), True)
        # test hmc inverse pde dictionary
        y = observe_mat @ u + np.random.randn((n // 2 + 1) ** 2) * inv_pde['noise_ob']
        inputs = np.array([0, 0, 1, -1, 0], dtype=np.float)
        hmc_inv_pde = hmc.compute_hmc_dictionary(inv_pde=inv_pde, observe_mat=observe_mat, y=y)
        print(hmc_inv_pde.keys())
        print(list(map(type, hmc_inv_pde.values())))
        self.assertEqual(np.allclose(hmc_inv_pde['observe_mat'].toarray(), observe_mat.toarray()), True)
        self.assertEqual(np.allclose(hmc_inv_pde['y'], y), True)
        # test compute potential
        loglik = hmc.compute_potential(inputs=inputs, hmc_inv_pde=hmc_inv_pde, order=0)
        dloglik = hmc.compute_potential(inputs=inputs, hmc_inv_pde=hmc_inv_pde, order=1)
        print(loglik, dloglik)

        # test hmc evolve
        theta_after_burnin, acp_num, timer, x_data, sol_data, sol_grad_data, potential_data = hmc.hmc_evolve(
            hmc_inv_pde=hmc_inv_pde, num_of_iter=50, state='burnin', start_theta=np.zeros(inputs.shape))
        training_size = int(sum(acp_num))
        self.assertEqual(theta_after_burnin.shape, inputs.shape)
        self.assertEqual(timer.shape, (len(acp_num),))
        self.assertEqual(x_data.shape, (kl_dim, training_size))
        self.assertEqual(sol_data.shape, ((n+1)**2, training_size))
        self.assertEqual(sol_grad_data.shape, ((n+1)**2, kl_dim, training_size))
        self.assertEqual(potential_data.shape, (1, training_size))



if __name__ == '__main__':
    unittest.main()
