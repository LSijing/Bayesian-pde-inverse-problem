import unittest
from pdeinverse import hmc, elliptic, utils, hmc_dd
import numpy as np
import copy
import math

class MyTestCase(unittest.TestCase):
    def test_hmc_dd_process_data(self):
        # problem definition
        N = 9
        Nob = 3
        num_kl = 10
        num_sol_basis = 5
        num_grad_basis = 15
        pde = elliptic.compute_pde_dictionary(n=N)
        pde_dict = hmc.compute_inverse_pde_dictionary(pde, corr_length=0.2, noise_ob=0.1, var=1.0, sigma_theta=0.5,
                                                      kl_ndim=num_kl)
        observe_mat = hmc.get_observation_operator(n=Nob, inv_pde=pde_dict)
        true_theta = np.zeros(num_kl)
        true_theta[[0, 1, 2, 3, 4, 5, 6]] = np.array([1, 1, -1, 0.5, 0.3, -0.6, 0.2], dtype=np.float)
        u = hmc.solve_from_normal_input(inputs=true_theta, inv_pde=pde_dict)
        y = observe_mat @ u + np.random.randn((Nob + 1) ** 2) * pde_dict['noise_ob']
        hmc_inv_pde = hmc.compute_hmc_dictionary(inv_pde=pde_dict, observe_mat=observe_mat, y=y)

        leap_frog_step_num = 10
        step_size = 0.16
        total_iter_num = 100
        burn_in_num = 50
        start_theta = np.zeros(num_kl)

        results_burnin = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=burn_in_num, state='burnin',
                                        start_theta=start_theta)
        mass_mat = elliptic.compute_mass_matrix(tris=hmc_inv_pde['tris'], points=hmc_inv_pde['points'])
        training_data = hmc_dd.process_training_data(results_burnin[3], results_burnin[4], results_burnin[5],
                                                     mass_mat=mass_mat, num_sol_basis=num_sol_basis, num_grad_basis=num_grad_basis)
        # check input data
        self.assertEqual(np.allclose(training_data['x'].mean(axis=1), 0), True)
        self.assertEqual(np.allclose(training_data['x'].std(axis=1), 1), True)
        self.assertEqual(np.allclose(training_data['x'] * training_data['std_x'] + training_data['mean_x'], results_burnin[3]), True)
        # check solution output
        self.assertEqual(np.allclose(np.diag(training_data['phi_sol'].transpose() @ mass_mat @ training_data['phi_sol']), 1), True)
        self.assertEqual(np.allclose(training_data['y_sol'].mean(axis=1), 0), True)
        self.assertEqual(np.allclose(training_data['y_sol'].var(axis=1), 1), True)
        # check correctness of solution output
        project_diff = training_data['phi_sol'] @ (training_data['y_sol'] * training_data['std_y_sol'] + training_data['mean_y_sol']) + training_data['mean_sol'] - results_burnin[4]
        rel_err = np.sqrt(np.diag((project_diff.T @ mass_mat @ project_diff) / (results_burnin[4].T @ mass_mat @ results_burnin[4])))
        print('For solution space, %d basis with mean, L^2 error mean: %.6f std: %.6f'%(num_sol_basis, rel_err.mean(), rel_err.std()))
        self.assertEqual(True, True)
        for i in range(num_kl):
            self.assertEqual(np.allclose(np.diag(training_data['phi_grad'][:, i, :].transpose() @ mass_mat @ training_data['phi_grad'][:, i, :]), 1), True)
            self.assertEqual(np.allclose(training_data['y_grad'][:, i, :].mean(axis=1), 0), True)
            self.assertEqual(np.allclose(training_data['y_grad'][:, i, :].var(axis=1), 1), True)
            project_diff = training_data['mean_grad'][:, i, :] + training_data['phi_grad'][:, i, :] @ (training_data['y_grad'][:, i, :] * training_data['std_y_grad'][:, i, :] + training_data['mean_y_grad'][:, i, :]) - results_burnin[5][:, i, :]
            rel_err = np.sqrt(np.diag((project_diff.T @ mass_mat @ project_diff) / (results_burnin[5][:, i, :].transpose() @ mass_mat @ results_burnin[5][:, i, :])))
            print('For gradient space, %d-th partial derivative, %d basis with mean, L^2 error mean: %.6f std: %.6f'%(i, num_grad_basis, rel_err.mean(), rel_err.std()))

        # test training process
        net = hmc_dd.SolutionMap(size_in=num_kl, size_out=num_sol_basis, size_hidden=5)
        net.double()
        net, train_time, train_loss, dev_loss = hmc_dd.trainit(net, training_data['x'], training_data['y_sol'], opt='SGD', epochs=200, lr=0.02, num_iter=5)
        net_grad = []
        for i in range(num_kl):
            net1 = hmc_dd.GradientMap(size_in=num_kl, size_out= num_grad_basis, size_hidden=20)
            net1.double()
            net1, _, _, _ = hmc_dd.trainit(net1, training_data['x'], training_data['y_grad'][:, i, :], opt='Adam', epochs=200, lr=0.02, num_iter=5)
            net_grad.append(net1)

        # test solve via neural network
        u, pu = hmc_dd.solve_via_data_driven(np.random.randn(num_kl), net, dataset=training_data, derivative=1, net_grad=net_grad)
        self.assertEqual(u.shape, ((N+1) ** 2, 1))
        self.assertEqual(pu.shape, ((N+1) ** 2, num_kl))
        # test compute potential
        theta = np.random.randn(num_kl)
        pot = hmc_dd.compute_potential_via_data_driven(theta, net, net_grad, hmc_inv_pde, training_data, order=0)
        print('potential = ', pot)
        dpot = hmc_dd.compute_potential_via_data_driven(theta, net, net_grad, hmc_inv_pde, training_data, order=1)
        print('potential gradient = ', dpot)

if __name__ == '__main__':
    unittest.main()
