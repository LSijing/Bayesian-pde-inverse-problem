from pdeinverse import hmc, elliptic, utils, hmc_dd
import numpy as np
import time

np.random.seed(123)
# problem definition
N = 30
Nob = 10
num_kl = 20
pde = elliptic.compute_pde_dictionary(n=N)
pde_dict = hmc.compute_inverse_pde_dictionary(pde, corr_length=0.2, noise_ob=0.1, var=1.0, sigma_theta=0.5, kl_ndim=num_kl)
observe_mat = hmc.get_observation_operator(n=Nob, inv_pde=pde_dict)
true_theta = np.zeros(num_kl)
true_theta[[0,1,2,3,4,5,6]] = np.array([1,1,-1,0.5,0.3,-0.6,0.2], dtype=np.float)
u = hmc.solve_from_normal_input(inputs=true_theta, inv_pde=pde_dict)
y = observe_mat @ u + np.random.randn((Nob + 1) ** 2) * pde_dict['noise_ob']
hmc_inv_pde = hmc.compute_hmc_dictionary(inv_pde=pde_dict, observe_mat=observe_mat, y=y)

leap_frog_step_num = 10
step_size = 0.16
total_iter_num = 50000
burn_in_num = 10000
start_theta = np.zeros(num_kl)
num_sol_basis = 20
num_grad_basis = 40

# burn-in, collect and process training data
theta_after_burnin, acp_num_burnin, timer_burnin, x_data, sol_data, sol_grad_data = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=burn_in_num, state='burnin', start_theta=start_theta)
mass_mat = elliptic.compute_mass_matrix(tris=hmc_inv_pde['tris'], points=hmc_inv_pde['points'])
training_data, basis_data = hmc_dd.process_training_data(x_data, sol_data, sol_grad_data, mass_mat=mass_mat, num_sol_basis=num_sol_basis, num_grad_basis=num_grad_basis)

# save burn-in data into '.npz' file for data-driven use
np.savez('burnin_data', theta_after_burnin=theta_after_burnin, acp_num_burnin=acp_num_burnin, timer_burnin=timer_burnin,
x_data=x_data, sol_data=sol_data, sol_grad_data=sol_grad_data, leap_frog_step_num=leap_frog_step_num, step_size=step_size,
burn_in_num=burn_in_num, num_sol_basis=num_sol_basis, num_grad_basis=num_grad_basis, basis_data=basis_data, training_data=training_data,
hmc_inv_pde=hmc_inv_pde)

# standard hmc
sampled_theta, acp_num, timer = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=total_iter_num, state='sample', start_theta=theta_after_burnin)
np.savez('hmc_data'+time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()), sampled_theta=sampled_theta, acp_num=acp_num, timer=timer)
