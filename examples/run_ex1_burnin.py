from pdeinverse import hmc, elliptic, utils, hmc_dd
from examples import example1
import numpy as np
import time

# define the problem and set algorithm parameters
num_kl = 25
N, Nob, num_kl, observe_mat, true_theta, u, y, hmc_inv_pde = example1.set_problem(num_kl=num_kl)
leap_frog_step_num, step_size, total_iter_num, burn_in_num, start_theta, num_sol_basis, num_grad_basis = example1.set_algorithm_settings(num_kl=num_kl)

# burn-in, collect and process training data
print('start burn-in')
theta_after_burnin, acp_num_burnin, timer_burnin, x_data, sol_data, sol_grad_data, potential_data = hmc.hmc_evolve(hmc_inv_pde=hmc_inv_pde, num_of_iter=burn_in_num, state='burnin', start_theta=start_theta)
mass_mat = elliptic.compute_mass_matrix(tris=hmc_inv_pde['tris'], points=hmc_inv_pde['points'])
training_data, basis_data = hmc_dd.process_training_data(x_data, sol_data, sol_grad_data, mass_mat=mass_mat, num_sol_basis=num_sol_basis, num_grad_basis=num_grad_basis)

# save burn-in data into '.npz' file for data-driven use
np.savez(f'burnin_data_{time.strftime("%Y_%m_%d_%H", time.gmtime())}', theta_after_burnin=theta_after_burnin, acp_num_burnin=acp_num_burnin, timer_burnin=timer_burnin,
x_data=x_data, sol_data=sol_data, sol_grad_data=sol_grad_data, potential_data=potential_data, leap_frog_step_num=leap_frog_step_num, step_size=step_size,
burn_in_num=burn_in_num, num_sol_basis=num_sol_basis, num_grad_basis=num_grad_basis, basis_data=basis_data, training_data=training_data,
hmc_inv_pde=hmc_inv_pde)

