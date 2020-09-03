from pdeinverse import elliptic, hmc, utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SolutionMap(nn.Module):
    def __init__(self, size_in=20, size_out=20, size_hidden=30):
        super(SolutionMap, self).__init__()
        self.size = size_hidden
        self.inp = nn.Linear(size_in, self.size)
        self.l1 = nn.Linear(self.size, self.size)
        self.l2 = nn.Linear(self.size, self.size)
        # self.l3 = nn.Linear(self.size, self.size)
        # self.l4 = nn.Linear(self.size,self.size)
        # self.l5 = nn.Linear(self.size,self.size)
        self.out = nn.Linear(self.size, size_out)
        self.acti = F.relu

    def forward(self, x):
        y = self.inp(x)
        y = self.acti(self.l1(y))
        # y = self.acti(self.l2(y)) + y
        # y = self.acti(self.l3(y)) + y
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
        self.acti = F.relu

    def forward(self, x):
        y = self.inp(x)
        y = self.acti(self.l1(y))
        y = self.acti(self.l2(y)) + y
        y = self.acti(self.l3(y)) + y
        y = self.out(y)
        return y

def process_training_data(x_data, sol_data, sol_grad_data, mass_mat, num_sol_basis, num_grad_basis):
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

    dataset = {'x': x, 'mean_x': mean_x, 'std_x': std_x, 'y_sol': y_sol, 'mean_y_sol': mean_y_sol, 'std_y_sol': std_y_sol,
               'y_grad': y_grad, 'mean_y_grad': mean_y_grad, 'std_y_grad': std_y_grad, 'mean_sol': mean_sol, 'phi_sol': phi_sol,
               'mean_grad': mean_grad, 'phi_grad': phi_grad}
    return dataset


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
            if (j%(epochs//2)==epochs//2-1):
                print('loss ### %.8f ### %.8f ### %.8f'%(loss.item(),train_loss_series[-1], dev_loss_series[-1]))

    return net, train_time_series, train_loss_series, dev_loss_series


def solve_via_data_driven(inputs, net: SolutionMap, dataset: dict, derivative=0, net_grad: list = None):
    x = torch.from_numpy(inputs)
    y = net(x).detach().numpy().reshape((-1, 1))
    u = dataset['phi_sol'] @ (y * dataset['std_y_sol'] + dataset['mean_y_sol']) + dataset['mean_sol']
    if derivative == 0:
        return u
    if derivative == 1:
        (n_sol, num_kl, num_grad_basis) = dataset['phi_grad'].shape
        pu = np.zeros((n_sol, num_kl))
        for i in range(num_kl):
            y_grad = net_grad[i](x).detach().numpy().reshape((-1, 1))
            du_i = dataset['phi_grad'][:, i, :] @ (y_grad * dataset['std_y_grad'][:, i, :] + dataset['mean_y_grad'][:, i, :]) + dataset['mean_grad'][:, i, :]
            pu[:, i] = du_i.flatten()
        return u, pu


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
