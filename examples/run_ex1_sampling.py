import numpy as np
import time
from pdeinverse import hmc_dd, hmc_rns, hmc
from examples import example1
import torch

# read burn-in data
data = example1.read_burnin('burnin_data_2021_01_07_04.npz')
(theta_after_burnin, x_data, potential_data, leap_frog_step_num, step_size,
 num_sol_basis, num_grad_basis, basis_data, training_data, hmc_inv_pde) = \
    data['theta_after_burnin'], data['x_data'], data['potential_data'], data['leap_frog_step_num'], \
    data['step_size'], data['num_sol_basis'], data['num_grad_basis'], \
    data['basis_data'], data['training_data'], data['hmc_inv_pde']
total_iter_num = 500

# train rns
s = 1000
dismissed = 1000
x_train_rns = x_data[:, dismissed:]
y_train_rns = potential_data[:, dismissed:] - np.min(potential_data[:, dismissed:])
rnet = hmc_rns.RandomNet(size_in=hmc_inv_pde['kl_ndim'], size_out=1, size_hidden=s)
rnet = hmc_rns.trainit(rnet, x_train=x_train_rns, y_train=y_train_rns, epsilon=1e-7, bias=False)

# train two neural networks
net = hmc_dd.SolutionMap(size_in=hmc_inv_pde['kl_ndim'], size_out=num_sol_basis, size_hidden=20)
net.double()
net, train_time, train_loss, dev_loss = hmc_dd.trainit(net, training_data['x'], training_data['y_sol'], opt='Adam', epochs=2000, lr=0.01, num_iter=5)
net_grad = hmc_dd.GradientMap(size_in=hmc_inv_pde['kl_ndim'], size_hidden=40, size_out=num_grad_basis*hmc_inv_pde['kl_ndim'])
net_grad.double()
net_grad, train_iter, train_lost, _ = hmc_dd.trainit(net_grad, training_data['x'], training_data['y_grad'].reshape((-1, training_data['y_grad'].shape[2]), order='F'),
                                                     opt='Adam', epochs=2000, lr=0.02, num_iter=5)

for _ in range(2):
    # standard hmc
    sampled_theta, acp_num, timer = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, state='sample', start_theta=theta_after_burnin)
    np.savez(f'hmc_data{time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())}', sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)

    # random network surrogate solver
    sampled_theta, acp_num, timer = hmc_rns.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num,
                                                       start_theta=theta_after_burnin, net=rnet, step_size=step_size, num_of_leap_frog_steps=leap_frog_step_num)
    np.savez(f'hmc_rns_data{time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())}', sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)

    # data-driven solver
    sampled_theta, acp_num, timer = hmc_dd.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, start_theta=theta_after_burnin, net=net, net_grad=net_grad,
                      basis_data=basis_data, step_size=step_size, num_of_leap_frog_steps=leap_frog_step_num)
    np.savez(f'hmc_dd_data{time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())}', sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)
