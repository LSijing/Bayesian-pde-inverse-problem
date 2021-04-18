from pdeinverse import hmc, elliptic, utils, hmc_dd
import numpy as np
import time


def set_problem(num_kl: int):
    N = 30
    Nob = 10
    pde = elliptic.compute_pde_dictionary(n=N)
    pde_dict = hmc.compute_inverse_pde_dictionary(pde, corr_length=0.2, noise_ob=0.1, var=1.0, sigma_theta=0.5,
                                                  kl_ndim=num_kl)
    observe_mat = hmc.get_observation_operator(n=Nob, inv_pde=pde_dict)
    true_theta = np.random.randn(num_kl)
    u = hmc.solve_from_normal_input(inputs=true_theta, inv_pde=pde_dict)
    y = observe_mat @ u + np.random.randn((Nob + 1) ** 2) * pde_dict['noise_ob']
    hmc_inv_pde = hmc.compute_hmc_dictionary(inv_pde=pde_dict, observe_mat=observe_mat, y=y)
    return N, Nob, num_kl, observe_mat, true_theta, u, y, hmc_inv_pde


def set_algorithm_settings(num_kl: int):
    leap_frog_step_num = 10
    step_size = 0.16
    total_iter_num = 200000
    burn_in_num = 10000
    start_theta = np.zeros(num_kl)
    num_sol_basis = 20
    num_grad_basis = 40
    return leap_frog_step_num, step_size, total_iter_num, burn_in_num, start_theta, num_sol_basis, num_grad_basis


def read_burnin(path: str) -> dict:
    f = np.load(path)
    data = dict()
    data['theta_after_burnin'] = f['theta_after_burnin']
    data['acp_num_burnin'] = f['acp_num_burnin']
    data['timer_burnin'] = f['timer_burnin']
    data['x_data'] = f['x_data']
    data['sol_data'] = f['sol_data']
    data['sol_grad_data'] = f['sol_grad_data']
    data['potential_data'] = f['potential_data']
    data['leap_frog_step_num'] = f['leap_frog_step_num'].item()
    data['step_size'] = f['step_size'].item()
    data['burn_in_num'] = f['burn_in_num'].item()
    data['num_sol_basis'] = f['num_sol_basis'].item()
    data['num_grad_basis'] = f['num_grad_basis'].item()
    data['basis_data'] = f['basis_data'].item()
    data['training_data'] = f['training_data'].item()
    data['hmc_inv_pde'] = f['hmc_inv_pde'].item()
    f.close()
    return data
