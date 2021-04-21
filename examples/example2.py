from pdeinverse import hmc, elliptic, utils, hmc_dd
import numpy as np
import time


def get_problem_settings(num_kl: int):
    N = 64
    Nob = 16
    pde = elliptic.compute_pde_dictionary(n=N)
    pde_dict = hmc.compute_inverse_pde_dictionary(pde, corr_length=0.2, noise_ob=0.1, var=1.0, sigma_theta=0.5, kl_ndim=num_kl, corr_hetero=(0.08, 0.4))
    observe_mat = hmc.get_observation_operator(n=Nob, inv_pde=pde_dict)
    true_theta = np.random.rand(num_kl)
    u = hmc.solve_from_normal_input(inputs=true_theta, inv_pde=pde_dict)
    y = observe_mat @ u + np.random.randn((Nob + 1) ** 2) * pde_dict['noise_ob']
    hmc_inv_pde = hmc.compute_hmc_dictionary(inv_pde=pde_dict, observe_mat=observe_mat, y=y)
    return N, Nob, num_kl, observe_mat, true_theta, u, y, hmc_inv_pde


def get_algorithm_settings(num_kl: int):
    leap_frog_step_num = 10
    step_size = 0.16
    total_iter_num = 200000
    burn_in_num = 10000
    start_theta = np.zeros(num_kl)
    num_sol_basis = 20
    num_grad_basis = 40
    return leap_frog_step_num, step_size, total_iter_num, burn_in_num, start_theta, num_sol_basis, num_grad_basis
