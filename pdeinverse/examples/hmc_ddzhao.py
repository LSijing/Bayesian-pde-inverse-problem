import numpy as np
import time
from pdeinverse import hmc_dd

# possible improvement
# 1. grid size, given corr-length, kl_ndim
# 2. small scale, i.e. corr-length decrease,
# 3. kl_ndim increases

f = np.load('burnin_data.npz')
np.random.seed(123)
theta_after_burnin = f['theta_after_burnin']
acp_num_burnin = f['acp_num_burnin']
timer_burnin = f['timer_burnin']
x_data = f['x_data']
sol_data = f['sol_data']
sol_grad_data = f['sol_grad_data']
leap_frog_step_num = f['leap_frog_step_num'].item()
step_size = f['step_size'].item()
burn_in_num = f['burn_in_num'].item()
num_sol_basis = f['num_sol_basis'].item()
num_grad_basis = f['num_grad_basis'].item()
basis_data = f['basis_data'].item()
training_data = f['training_data'].item()
hmc_inv_pde = f['hmc_inv_pde'].item()
f.close()

# train data-driven solver
net = hmc_dd.SolutionMap(size_in=hmc_inv_pde['kl_ndim'], size_out=num_sol_basis, size_hidden=20)
net.double()
net, train_time, train_loss, dev_loss = hmc_dd.trainit(net, training_data['x'], training_data['y_sol'], opt='Adam', epochs=2000, lr=0.1, num_iter=6)
net_grad = []
# for i in range(num_kl):
#     print('------------------------------\n start training', i, '-th partial derivative')
#     net1 = hmc_dd.GradientMap(size_in=num_kl, size_out= num_grad_basis, size_hidden=30)
#     net1.double()
#     net1, _, _, _ = hmc_dd.trainit(net1, training_data['x'], training_data['y_grad'][:, i, :], opt='Adam', epochs=500, lr=0.02, num_iter=5)
#     net_grad.append(net1)


# hmc_dd.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, start_theta=theta_after_burnin, net=net, net_grad=net_grad,
#                   basis_data=basis_data, step_size=step_size, num_of_leap_frog_steps=leap_frog_step_num)

# np.savez('ex1', sampled_theta=, theta_netSaved=theta_netSaved, IterNum=IterNum, BurnIn=BurnIn,
#         acp_net=acp_net, timer_net=timer_net, acp=acp, timer=timer,
#         num_basis_grad=num_basis_grad, num_basis_sol=num_basis_sol, num_kl=num_kl )

# np.load('burnin_data.npz')