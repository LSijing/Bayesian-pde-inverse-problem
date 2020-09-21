import unittest
from pdeinverse import hmc_rns, hmc, elliptic
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_initialization(self):
        sizes = [2, 1000, 1]
        net = hmc_rns.RandomNet(size_in=sizes[0], size_hidden=sizes[1], size_out=sizes[2])
        self.assertEqual(net.W1.shape, (sizes[1], sizes[0]))
        self.assertEqual(net.b1.shape, (sizes[1], 1))
        self.assertEqual(net.W2.shape, (sizes[2], sizes[1]))
        self.assertEqual(net.b2.shape, (sizes[2], 1))
        n = 10000
        x = np.random.randn(sizes[0], n)
        z1 = net.W1 @ x + net.b1
        a1 = net.activation(z1)
        y = net.predict(x)
        # print(z1.mean(), z1.var())
        # print(a1.mean(), a1.var())
        # print(y.mean(), y.var())
        # for 1-d np array
        x2 = np.random.randn(sizes[0])
        y2 = net.predict(x2.reshape((sizes[0], 1)))
        y_true = net.W2 @ net.activation(net.W1 @ x2.reshape((sizes[0], 1)) + net.b1) + net.b2
        self.assertEqual(y2.shape, (1, 1))
        self.assertEqual(np.allclose(y2, y_true), True)

    def test_gradient(self):
        sizes = [2, 1000, 1]
        net = hmc_rns.RandomNet(size_in=sizes[0], size_hidden=sizes[1], size_out=sizes[2])
        x = np.random.randn(sizes[0], 1)
        dx = 0.001
        dx1 = np.array([[dx], [0.]])
        dx2 = np.array([[0.], [dx]])
        y = net.predict(x)
        dy1 = net.predict(x + dx1) - y
        dy2 = net.predict(x + dx2) - y
        grad = net.grad_predict(x)
        self.assertEqual(np.allclose(grad.T @ dx1 - dy1, 0, atol=dx**2), True)
        self.assertEqual(np.allclose(grad.T @ dx2 - dy2, 0, atol=dx**2), True)

        sizes = [20, 100, 1]
        net2 = hmc_rns.RandomNet(size_in=sizes[0], size_hidden=sizes[1], size_out=sizes[2])
        x = np.random.randn(sizes[0], 1)
        dx = np.random.randn(sizes[0], 1) * 0.0001
        dy = net2.predict(x + dx) - net2.predict(x)
        grad2 = net2.grad_predict(x)
        self.assertEqual(np.allclose(grad2.T @ dx - dy, 0), True)

    def test_trainit(self):
        sizes = [2, 10, 1]
        net = hmc_rns.RandomNet(size_in=sizes[0], size_hidden=sizes[1], size_out=sizes[2])
        # training samples
        m = 100
        x = np.random.randn(sizes[0], m)
        z1 = net.W1 @ x + net.b1
        a1 = net.activation(z1)
        coef = np.random.randn(sizes[2], sizes[1])
        bias = np.random.randn(sizes[2], 1)
        noise = 0.001
        y = coef @ a1 + bias + np.random.randn(1, m) * noise
        # test samples
        x_test = np.random.randn(sizes[0], m)
        z1 = net.W1 @ x_test + net.b1
        a1 = net.activation(z1)
        y_test = coef @ a1 + bias + np.random.randn(1, m) * noise
        # training process
        net = hmc_rns.trainit(net, x_train=x, y_train=y, epsilon=1e-6)
        y_bar = net.predict(x)
        rss = np.sum((y_bar - y) ** 2) / m
        # test process
        y_test_bar = net.predict(x_test)
        self.assertEqual(y_test_bar.shape, (1, m))
        rss_test = np.sum((y_test_bar - y_test) ** 2) / m
        # print(rss)
        # print(rss_test)

    def test_hmc_evolve(self):
        # problem definition
        N = 9
        Nob = 3
        num_kl = 10
        num_sol_basis = 5
        num_grad_basis = 15
        pde = elliptic.compute_pde_dictionary(n=N)
        pde_dict = hmc.compute_inverse_pde_dictionary(pde, corr_length=0.5, noise_ob=0.1, var=1.0, sigma_theta=0.5,
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

        theta_after_burnin, acp_num, timer, x_data, _, _, potential_data = hmc.hmc_evolve(
            hmc_inv_pde=hmc_inv_pde, num_of_iter=burn_in_num, state='burnin', start_theta=start_theta)

        s = 10
        net = hmc_rns.RandomNet(size_in=num_kl, size_hidden=s, size_out=1)
        hmc_rns.trainit(net, x_train=x_data, y_train=potential_data, epsilon=1e-6)
        rss = np.sum((net.predict(x_data) - potential_data) ** 2) / timer.size
        print('training residual sum of squares =', rss)

        theta_samples, acp, timer = hmc_rns.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num,
                                                      start_theta=theta_after_burnin, net=net, step_size=step_size,
                                                      num_of_leap_frog_steps=leap_frog_step_num)
        self.assertEqual(theta_samples.shape, (num_kl, total_iter_num))
        self.assertEqual(timer.size, total_iter_num)
        self.assertEqual(len(acp), total_iter_num)

if __name__ == '__main__':
    unittest.main()
