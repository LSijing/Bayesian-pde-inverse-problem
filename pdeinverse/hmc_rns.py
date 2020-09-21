import numpy as np
from pdeinverse import hmc
import copy
import math
import time


class RandomNet:
    """
    neural network with one single hidden layer
    input:
        x: column-wise data
    forward:
        z1 = W1 @ x + b1
        a1 = activation(z1)
        z2 = W2 @ a1 + b2
        y_bar = z2
    """

    def __init__(self, size_in: int, size_out: int, size_hidden: int):
        self.W1 = np.random.randn(size_hidden, size_in) / np.sqrt(size_in)
        self.b1 = np.random.randn(size_hidden, 1)
        self.W2 = np.random.randn(size_out, size_hidden) / np.sqrt(size_hidden)
        self.b2 = np.random.randn(size_out, 1)

    @staticmethod
    def activation(z):
        return np.log(1. + np.exp(z))

    @staticmethod
    def activation_derivative_1(z):
        return 1. / (1. + np.exp(-z))

    def predict(self, x):
        a1 = self.activation(self.W1 @ x + self.b1)
        y_bar = self.W2 @ a1 + self.b2
        return y_bar

    def grad_predict(self, x):
        a_prime = self.activation_derivative_1(self.W1 @ x + self.b1)
        grad = ((self.W2 * a_prime.T) @ self.W1).reshape((-1, 1))
        return grad


def trainit(net: RandomNet, x_train: np.ndarray, y_train: np.ndarray, bias=True, epsilon=1e-8):
    z1 = net.W1 @ x_train + net.b1
    a1 = net.activation(z1)
    n, m = z1.shape
    if bias:
        feature_matrix = np.concatenate((np.ones((1, m)), a1), axis=0)
        assert feature_matrix.shape == (n + 1, m)
        ols_sol = np.linalg.solve(feature_matrix @ feature_matrix.T + epsilon * np.eye(n + 1), feature_matrix @ y_train.T)
        assert ols_sol.shape == (n + 1, 1)
        net.b2 = ols_sol[0:1, :].T
        net.W2 = ols_sol[1:, :].T
    else:
        feature_matrix = a1
        assert feature_matrix.shape == (n, m)
        ols_sol = np.linalg.solve(feature_matrix @ feature_matrix.T + epsilon * np.eye(n), feature_matrix @ y_train.T)
        net.W2 = ols_sol.T
    return net


def hmc_evolve(hmc_inv_pde: dict, num_of_iter: int, start_theta, net: RandomNet,
               step_size: float = 0.16, num_of_leap_frog_steps: int = 10):
    """
    HMC evolution using random network surrogate
    """
    timer = np.zeros(num_of_iter)
    acp_num = []
    n = hmc_inv_pde['kl_ndim']
    sampled_theta = np.zeros((n, num_of_iter))
    current_theta = copy.copy(start_theta)
    current_potential = hmc.compute_potential(current_theta, hmc_inv_pde, order=0)
    for i in range(num_of_iter):
        time_start = time.clock()
        # propose theta, momentum
        proposed_theta = copy.copy(current_theta)
        current_momentum = np.random.randn(n)
        proposed_momentum = copy.copy(current_momentum)

        # Hamiltonian dynamics leap frog evolution
        random_num_leap_frog = np.random.choice(num_of_leap_frog_steps) + 1
        for j in range(random_num_leap_frog):
            proposed_momentum -= step_size / 2 * (net.grad_predict(proposed_theta.reshape((-1, 1)))).flatten()
            proposed_theta += step_size * proposed_momentum
            proposed_momentum -= step_size / 2 * (net.grad_predict(proposed_theta.reshape((-1, 1)))).flatten()

        # compute potentials
        proposed_potential = hmc.compute_potential(proposed_theta, hmc_inv_pde, order=0)

        # compute Hamiltonian
        proposed_H = proposed_potential + 0.5 * (proposed_momentum ** 2).sum()
        current_H = current_potential + 0.5 * (current_momentum ** 2).sum()
        ratio = current_H - proposed_H

        # accept reject
        if math.isfinite(ratio) and np.log(np.random.uniform()) < ratio:
            current_theta = proposed_theta
            current_potential = proposed_potential
            acp_num.append(1)
        else:
            acp_num.append(0)
        timer[i] = time.clock() - time_start
        sampled_theta[:, i] = current_theta

        if (i + 1) % 100 == 0:
            print('finished %d iterations.' % (i + 1))
            print('acceptance rate of latest 100 iterations %.2f' % (sum(acp_num[-100:]) / 100))

    return sampled_theta, acp_num, timer
