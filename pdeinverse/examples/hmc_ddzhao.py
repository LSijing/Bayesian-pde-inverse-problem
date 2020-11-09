import numpy as np
import time
from pdeinverse import hmc_dd, hmc_rns, hmc
import torch

# possible improvement
# 1. grid size, given corr-length, kl_ndim
# 2. small scale, i.e. corr-length decrease,
# 3. kl_ndim increases

f = np.load('burnin_data_2020_09_23_09.npz')
theta_after_burnin = f['theta_after_burnin']
acp_num_burnin = f['acp_num_burnin']
timer_burnin = f['timer_burnin']
x_data = f['x_data']
sol_data = f['sol_data']
sol_grad_data = f['sol_grad_data']
potential_data = f['potential_data']
leap_frog_step_num = f['leap_frog_step_num'].item()
step_size = f['step_size'].item()
burn_in_num = f['burn_in_num'].item()
num_sol_basis = f['num_sol_basis'].item()
num_grad_basis = f['num_grad_basis'].item()
basis_data = f['basis_data'].item()
training_data = f['training_data'].item()
hmc_inv_pde = f['hmc_inv_pde'].item()
f.close()
total_iter_num = 50000

for _ in range(5):
    # standard hmc
    sampled_theta, acp_num, timer = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, state='sample', start_theta=theta_after_burnin)
    np.savez('hmc_data'+time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()), sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)


    # random network surrogate solver
    s = 1000
    dismissed = 1000
    x_train_rns = x_data[:, dismissed:]
    y_train_rns = potential_data[:, dismissed:] - np.min(potential_data[:, dismissed:])
    rnet = hmc_rns.RandomNet(size_in=hmc_inv_pde['kl_ndim'], size_out=1, size_hidden=s)
    rnet = hmc_rns.trainit(rnet, x_train=x_train_rns, y_train=y_train_rns, epsilon=1e-7, bias=False)
    sampled_theta, acp_num, timer = hmc_rns.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num,
                                                       start_theta=theta_after_burnin, net=rnet, step_size=step_size, num_of_leap_frog_steps=leap_frog_step_num)
    np.savez('hmc_rns_data'+time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()), sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)

    # train data-driven solver
    # torch.manual_seed(12)
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
