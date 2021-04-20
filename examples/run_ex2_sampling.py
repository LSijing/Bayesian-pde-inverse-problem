import numpy as np
import time
from pdeinverse import hmc_dd, hmc_rns, hmc
from examples import example1
import torch

# read burn-in data
data = example1.read_burnin('burnin_data_2021_01_22_18.npz')
(theta_after_burnin, x_data, potential_data, leap_frog_step_num, step_size,
 num_sol_basis, num_grad_basis, basis_data, training_data, hmc_inv_pde) = \
    data['theta_after_burnin'], data['x_data'], data['potential_data'], data['leap_frog_step_num'], \
    data['step_size'], data['num_sol_basis'], data['num_grad_basis'], \
    data['basis_data'], data['training_data'], data['hmc_inv_pde']
total_iter_num = 100000

# data-driven HMC
net = hmc_dd.SolutionMap(size_in=hmc_inv_pde['kl_ndim'], size_out=num_sol_basis, size_hidden=20)
net.double()
net, train_time, train_loss, dev_loss = hmc_dd.trainit(net, training_data['x'], training_data['y_sol'], opt='Adam', epochs=2000, lr=0.01, num_iter=5)
net_grad = hmc_dd.GradientMap(size_in=hmc_inv_pde['kl_ndim'], size_hidden=40, size_out=num_grad_basis*hmc_inv_pde['kl_ndim'])
net_grad.double()
net_grad, train_iter, train_lost, _ = hmc_dd.trainit(net_grad, training_data['x'], training_data['y_grad'].reshape((-1, training_data['y_grad'].shape[2]), order='F'),
                                                     opt='Adam', epochs=2000, lr=0.02, num_iter=5)

sampled_theta, acp_num, timer = hmc_dd.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, start_theta=theta_after_burnin, net=net, net_grad=net_grad,
                  basis_data=basis_data, step_size=step_size, num_of_leap_frog_steps=leap_frog_step_num)
np.savez('hmc_dd_data'+time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()), sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)

# standard hmc
sampled_theta, acp_num, timer = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, state='sample', start_theta=theta_after_burnin)
np.savez('hmc_data'+time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()), sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)
