from pdeinverse import elliptic, hmc, utils
import numpy as np
import scipy
from scipy.sparse.linalg import spsolve
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time

class SolutionMap(nn.Module):
    def __init__(self, size_in=20, size_out=20, size_hidden=30):
        super(SolutionMap, self).__init__()
        self.size = size_hidden
        self.inp = nn.Linear(size_in, self.size)
        self.l1 = nn.Linear(self.size, self.size)
        self.l2 = nn.Linear(self.size, self.size)
        self.l3 = nn.Linear(self.size, self.size)
        # self.l4 = nn.Linear(self.size,self.size)
        # self.l5 = nn.Linear(self.size,self.size)
        self.out = nn.Linear(self.size, size_out)
        self.acti = F.tanh

    def forward(self, x):
        y = self.inp(x)
        y = self.acti(self.l1(y)) + y
        y = self.acti(self.l2(y)) + y
        y = self.acti(self.l3(y)) + y
        # y = self.acti(self.l4(y))+y
        # y = self.acti(self.l5(y))+y
        y = self.out(y)
        return y


class GradientMap(nn.Module):
    def __init__(self, size_in=10, size_out=20, size_hidden=30):
        super(GradientMap, self).__init__()
        self.size = size_hidden
        self.inp = nn.Linear(size_in, size_hidden)
        self.l1 = nn.Linear(size_hidden, size_hidden)
        self.l2 = nn.Linear(size_hidden, size_hidden)
        self.l3 = nn.Linear(size_hidden, size_hidden)
        self.out = nn.Linear(size_hidden, size_out)
        self.acti = F.tanh

    def forward(self, x):
        y = self.inp(x)
        y = self.acti(self.l1(y)) + y
        y = self.acti(self.l2(y)) + y
        y = self.acti(self.l3(y)) + y
        y = self.out(y)
        return y

def process_training_data(x_data, sol_data, sol_grad_data, mass_mat, num_sol_basis, num_grad_basis, grad_together=False):
    """
    input:
        x_data: (n_input, m), m samples, n_input dimension each
        sol_data: (n_sol, m), m samples, n_sol dimension each
        sol_grad_data: (n_sol, n_input, m), m samples, n_input partial derivatives, n_sol dimension each
    output:
        x: (n_input, m) with mean=0, var=1
        y_sol: (num_basis, m) with mean=0, var=1
        y_grad: (num_basis_grad, n_input, m) with mean=0, var=1 for each n_input
        mean_x, std_x, mean_y_sol, std_y_sol, mean_y_grad, sta_y_grad:

        mean_sol: (n_sol, 1)
        phi_sol: (n_sol, num_sol_basis)
        mean_grad: (n_sol, n_input, 1)
        phi_grad: (n_sol, n_input, num_grad_basis)
    """
    def compute_mean_std(data: np.ndarray):
        """
        data: column-wise data
        """
        n, m = data.shape
        mean = np.mean(data, axis=1, keepdims=True)
        assert mean.shape == (n, 1)
        std = np.var(data, axis=1, keepdims=True) ** 0.5
        assert std.shape == (n, 1)
        x = (data - mean) / std
        return x, mean, std

    # degree of freedoms
    n_input, m = x_data.shape
    n_sol = sol_data.shape[0]

    # input data
    x, mean_x, std_x = compute_mean_std(x_data)
    # print(x.shape, x_data.shape)
    assert x.shape == (n_input, m)
    assert mean_x.shape == (n_input, 1)
    assert std_x.shape == (n_input, 1)

    # solution output (bases and coefficients)
    phi_sol, _ = utils.compute_PCA(sol_data, mean=True, k=num_sol_basis, A=mass_mat, normalize=True)
    mean_sol, phi_sol = phi_sol[:, 0].reshape((-1, 1)), phi_sol[:, 1:]
    y_sol = phi_sol.transpose() @ mass_mat @ (sol_data - mean_sol)
    y_sol, mean_y_sol, std_y_sol = compute_mean_std(y_sol)
    assert y_sol.shape == (num_sol_basis, m)
    assert mean_y_sol.shape == (num_sol_basis, 1)
    assert std_y_sol.shape == (num_sol_basis, 1)
    assert mean_sol.shape == (n_sol, 1)
    assert phi_sol.shape == (n_sol, num_sol_basis)

    # gradient output (bases and coefficients, each partial derivative)
    if not grad_together:
        y_grad = np.zeros((num_grad_basis, n_input, m))
        mean_y_grad = np.zeros((num_grad_basis, n_input, 1))
        std_y_grad = np.zeros((num_grad_basis, n_input, 1))
        mean_grad, phi_grad = np.zeros((n_sol, n_input, 1)), np.zeros((n_sol, n_input, num_grad_basis))
        for i in range(n_input):
            phi_grad_i, _ = utils.compute_PCA(sol_grad_data[:, i, :], mean=True, k=num_grad_basis, A=mass_mat, normalize=True)
            assert phi_grad_i.shape == (n_sol, num_grad_basis+1)
            mean_grad[:, i, :], phi_grad[:, i, :] = phi_grad_i[:, 0].reshape((-1, 1)), phi_grad_i[:, 1:]
            y_grad_i = phi_grad[:, i, :].transpose() @ mass_mat @ (sol_grad_data[:, i, :] - mean_grad[:, i, :])
            y_grad_i, mean_grad_i, std_grad_i = compute_mean_std(y_grad_i)
            y_grad[:, i, :] = y_grad_i
            mean_y_grad[:, i, :] = mean_grad_i
            std_y_grad[:, i, :] = std_grad_i
        assert y_grad.shape == (num_grad_basis, n_input, m)
        assert mean_y_grad.shape == (num_grad_basis, n_input, 1)
        assert std_y_grad.shape == (num_grad_basis, n_input, 1)
        assert mean_grad.shape == (n_sol, n_input, 1)
        assert phi_grad.shape == (n_sol, n_input, num_grad_basis)
    if grad_together:
        y_grad = np.zeros((num_grad_basis, n_input, m))
        mean_y_grad = np.zeros((num_grad_basis, n_input, 1))
        std_y_grad = np.zeros((num_grad_basis, n_input, 1))
        phi_grad_prime, _ = utils.compute_PCA(sol_grad_data.reshape((n_sol, -1)), mean=True, k=num_grad_basis, A=mass_mat,
                                          normalize=True)
        assert phi_grad_prime.shape == (n_sol, num_grad_basis + 1)
        phi_grad = phi_grad_prime[:, 1:]
        mean_grad = phi_grad_prime[:, 0:1]
        for i in range(n_input):
            y_grad_i = phi_grad.transpose() @ mass_mat @ (sol_grad_data[:, i, :] - mean_grad)
            y_grad_i, mean_grad_i, std_grad_i = compute_mean_std(y_grad_i)
            y_grad[:, i, :] = y_grad_i
            mean_y_grad[:, i, :] = mean_grad_i
            std_y_grad[:, i, :] = std_grad_i
        assert y_grad.shape == (num_grad_basis, n_input, m)
        assert mean_y_grad.shape == (num_grad_basis, n_input, 1)
        assert std_y_grad.shape == (num_grad_basis, n_input, 1)
        assert mean_grad.shape == (n_sol, 1)
        assert phi_grad.shape == (n_sol, num_grad_basis)

    training_set = {'x': x, 'y_sol': y_sol, 'y_grad': y_grad}
    basis_set = {'mean_x': mean_x, 'std_x': std_x, 'mean_y_sol': mean_y_sol, 'std_y_sol': std_y_sol,
                 'mean_y_grad': mean_y_grad, 'std_y_grad': std_y_grad, 'mean_sol': mean_sol, 'phi_sol': phi_sol,
                 'mean_grad': mean_grad, 'phi_grad': phi_grad}
    return training_set, basis_set


def trainit(net, x_train, y_train, opt='Adam', epochs=100, lr=0.02, num_iter=5):
    """
    input:
        net: an instance of SolutionMap()
        x_train, y_train: column-wise data
        num_iter: halving learning rate (num_iter-1) times
        epochs: training times with one specific learning rate
    output:
        trained NN, loss series
    """
    m = x_train.shape[1]
    ratio = 0.9
    shuffled = np.random.permutation(m)
    train_index, dev_index = shuffled[0: int(m * ratio)], shuffled[int(m * ratio):]
    x_data = torch.from_numpy(x_train.T)
    y_data = torch.from_numpy(y_train.T)
    train_x = x_data[train_index, :]
    train_y = y_data[train_index, :]
    dev_x = x_data[dev_index, :]
    dev_y = y_data[dev_index, :]
    assert train_x.shape[0] + dev_x.shape[0] == m

    learning_rate = lr
    criterion = nn.MSELoss()
    train_time_series = np.zeros(0)
    train_loss_series = np.zeros(0)
    dev_loss_series = np.zeros(0)

    for i in range(num_iter):
        if opt == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate / (2 ** i))
        elif opt == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate / (2 ** i))
        print(i + 1, 'out of', num_iter, 'learning rate = ', learning_rate / (2 ** i))
        # Adam(net.parameters(),lr = learning_rate/(4**i))
        for j in range(epochs):
            x = train_x
            y = train_y
            yout = net(x)
            loss = criterion(yout, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_time_series = np.concatenate([train_time_series, np.array(i * epochs + j + 1).reshape(1)])
            train_loss_series = np.concatenate((train_loss_series, ((net(train_x) - train_y) ** 2).mean(1).mean().detach().numpy().reshape(1)), 0)
            dev_loss_series = np.concatenate((dev_loss_series, ((net(dev_x) - dev_y) ** 2).mean(1).mean().detach().numpy().reshape(1)), 0)
            if (j%(epochs//5)==epochs//5-1):
                print('%d, loss ### %.8f ### %.8f ### %.8f'%(i*epochs+j+1, loss.item(),train_loss_series[-1], dev_loss_series[-1]))

    return net, train_time_series, train_loss_series, dev_loss_series


def solve_via_data_driven(inputs, net: SolutionMap, dataset: dict, derivative=0, net_grad: GradientMap = None):
    x = torch.from_numpy((inputs - dataset['mean_x'].flatten()) / dataset['std_x'].flatten())
    y = net(x).detach().numpy().reshape((-1, 1))
    u = dataset['phi_sol'] @ (y * dataset['std_y_sol'] + dataset['mean_y_sol']) + dataset['mean_sol']
    if derivative == 0:
        return u
    if derivative == 1:
        (n_sol, num_kl, num_grad_basis) = dataset['phi_grad'].shape
        pu = np.zeros((n_sol, num_kl))
        y_grad = net_grad(x).detach().numpy().reshape((num_grad_basis, num_kl), order='F')
        for i in range(num_kl):
            du_i = dataset['phi_grad'][:, i, :] @ (y_grad[:, i:i+1] * dataset['std_y_grad'][:, i, :] + dataset['mean_y_grad'][:, i, :]) + dataset['mean_grad'][:, i, :]
            pu[:, i] = du_i.flatten()
        return u, pu


def process_basis_data(basis_data: dict, mass_mat: np.ndarray):
    new_set = dict()
    keys_to_modify = ['mean_grad', 'phi_grad']
    for key in basis_data.keys():
        if key not in keys_to_modify:
            new_set[key] = basis_data[key]

    n_sol, num_grad_basis = basis_data['mean_grad'].shape[0], basis_data['mean_grad'].shape[2]
    new_set['mean_grad'] = np.mean(basis_data['mean_grad'], axis=1)
    assert new_set['mean_grad'].shape == (n_sol, 1)
    new_set['phi_grad'] = utils.compute_PCA(basis_data['phi_grad'].reshape((n_sol, -1), order='F'),
                                               mean=False, k=num_grad_basis, A=mass_mat)[0][:, 1:]
    assert new_set['phi_grad'].shape == (n_sol, num_grad_basis)
    return new_set

def solve_via_Galerkin(inputs, inv_pde: dict, dataset: dict, derivative=0):
    tris, points = inv_pde['tris'], inv_pde['points']
    free_node, fixed_node = inv_pde['free_node'], inv_pde['fixed_node']
    kl_ndim = inv_pde['kl_ndim']
    kl_modes = inv_pde['kl_modes']
    kl_weights = inv_pde['kl_weights']
    c = np.exp(((inputs * kl_weights) * kl_modes).sum(1))
    stiff_mat = elliptic.compute_stiffness_matrix(tris=tris, points=points, coef_discrete_on_center=c)
    A11 = stiff_mat.tocsr()[free_node, :][:, free_node]
    n = free_node.size + fixed_node.size
    u = np.zeros(n)
    u[fixed_node] = inv_pde['g_D']
    u[free_node] = spsolve(A11, (-stiff_mat.tocsr() @ u)[free_node])
    if derivative == 0:
        return u
    if derivative == 1:
        pu = np.zeros((n, kl_ndim))
        A_reduced = dataset['phi_grad'][free_node, :].transpose() @ A11 @ dataset['phi_grad'][free_node, :]
        for i in range(kl_ndim):
            pc = c * kl_weights[i] * kl_modes[:, i]
            pA = elliptic.compute_stiffness_matrix(tris=tris, points=points, coef_discrete_on_center=pc)
            pbLag = -pA.tocsr() @ u
            #            pu[:,i] = spsolve(A.tocsr()+self.robin_mat.tocsr(), pbLag.reshape((-1,1)) + self.robin_vec)
            pu[:, i] = dataset['mean_grad'].flatten() + dataset['phi_grad'] @ scipy.linalg.solve(A_reduced, dataset['phi_grad'][free_node, :].transpose() @ pbLag[free_node])
        return u, pu


def compute_potential_via_Galerkin(inputs):
    pass



def compute_potential_via_data_driven(inputs, net: SolutionMap, net_grad: list, hmc_inv_pde: dict, dataset: dict, order=0, sol_required=False):
    y = hmc_inv_pde['y']
    observe_mat = hmc_inv_pde['observe_mat']
    sigma_y, sigma_theta = hmc_inv_pde['noise_ob'], hmc_inv_pde['sigma_theta']
    u, pu = solve_via_data_driven(inputs, net, dataset, derivative=1, net_grad=net_grad)
    s = (observe_mat @ u).flatten()
    ds = observe_mat @ pu
    if order == 0:
        loglik = -((y - s) ** 2).sum() / (2 * sigma_y ** 2)
        logpri = - (inputs ** 2).sum() / (2 * sigma_theta ** 2)
        if sol_required:
            return -loglik - logpri, u, pu
        return -loglik - logpri
    elif order == 1:
        assert y.shape == s.shape
        dloglik = (np.tile((y - s).reshape((-1, 1)), (1, hmc_inv_pde['kl_ndim'])) * ds).sum(0) / (sigma_y ** 2)
        dlogpri = - inputs / (sigma_theta ** 2)
        if sol_required:
            return -dloglik - dlogpri, u, pu
        return -dloglik - dlogpri


def hmc_evolve(hmc_inv_pde: dict, num_of_iter: int, start_theta, net: SolutionMap, net_grad: list, basis_data: dict,
               step_size: float = 0.16, num_of_leap_frog_steps: int = 10):
    """
    HMC evolution using data-driven solver
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
            proposed_momentum -= step_size / 2 * compute_potential_via_data_driven(proposed_theta, net, net_grad, hmc_inv_pde, basis_data, order=1)
            proposed_theta += step_size * proposed_momentum
            proposed_momentum -= step_size / 2 * compute_potential_via_data_driven(proposed_theta, net, net_grad, hmc_inv_pde, basis_data, order=1)

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


