# -*- coding: utf-8 -*-

from pdeinverse import elliptic
import numpy as np
import copy

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import math
import time


def compute_inverse_pde_dictionary(pde: dict, var, corr_length, noise_ob, sigma_theta, kl_ndim, corr_hetero=None):
    """
        Gaussian process with zero mean, Gaussian kernel.
        cov(x1,x2) = var^2 * exp{-||x1-x2||^2/(2*corr_length^2)}
        kl_ndim terms of kl expansion modes

        output:
        a dictionary with kl modes,weights,dimensions, observation noise, prior variance
    """
    (n, d) = pde['center'].shape
    tmp = np.zeros((n, n))
    for k in range(d):
        pk = pde['center'][:, k:k + 1]
        if corr_hetero is not None:
            tmp = tmp + (np.tile(pk, (1, n)) - np.tile(pk.reshape((1, -1)), (n, 1))) ** 2 / (2 * corr_hetero[k] ** 2)
        else:
            tmp = tmp + (np.tile(pk, (1, n)) - np.tile(pk.reshape((1, -1)), (n, 1))) ** 2 / (2 * corr_length ** 2)
    cov = var ** 2 * np.exp(-tmp)
    w, v = np.linalg.eigh(cov)
    w = w[::-1]
    v = v[:, ::-1]
    kl_modes = v[:, 0:kl_ndim]
    kl_weights = np.sqrt(np.absolute(w[0:kl_ndim]))
    inv_pde = {'kl_modes': kl_modes, 'kl_weights': kl_weights, 'kl_ndim': kl_ndim, 'noise_ob': noise_ob,
               'sigma_theta': sigma_theta}
    inv_pde.update(pde)
    return inv_pde


def solve_from_normal_input(inputs, inv_pde: dict, derivative=0):
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
        for i in range(kl_ndim):
            pc = c * kl_weights[i] * kl_modes[:, i]
            pA = elliptic.compute_stiffness_matrix(tris=tris, points=points, coef_discrete_on_center=pc)
            pbLag = -pA.tocsr() @ u
            #            pu[:,i] = spsolve(A.tocsr()+self.robin_mat.tocsr(), pbLag.reshape((-1,1)) + self.robin_vec)
            pu[free_node, i] = spsolve(A11, pbLag[free_node])
        return u, pu


def get_observation_operator(n: int, inv_pde: dict):
    """
    """
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    [X, Y] = np.meshgrid(x, y)
    points = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    # only observe away from source, exclude points near the RHS source
    #        points = points[((points - np.array([0.3,0.3]))**2).sum(1)>0.01]
    #        points = points[((points - np.array([0.7,0.3]))**2).sum(1)>0.01]
    #        points = points[((points - np.array([0.3,0.7]))**2).sum(1)>0.01]
    #        points = points[((points - np.array([0.7,0.7]))**2).sum(1)>0.01]
    col_idx = np.array([np.where(np.absolute(points[i, :] - inv_pde['points']).sum(1) < 1e-8)[0] for i in
                        range(points.shape[0])]).flatten()
    #        Jsparse = np.array([ np.where((points[i,] == self.p).all(1))[0] for i in range(points.shape[0]) ]).flatten()
    observe_mat = coo_matrix((np.ones(points.shape[0]), (np.arange(points.shape[0]), col_idx)),
                             shape=(points.shape[0], inv_pde['points'].shape[0]))
    print('HMC : finished setting observation operator')
    return observe_mat


def compute_hmc_dictionary(inv_pde: dict, observe_mat, y):
    hmc_inv_pde = {'observe_mat': observe_mat, 'y': y}
    hmc_inv_pde.update(inv_pde)
    return hmc_inv_pde


def compute_potential(inputs, hmc_inv_pde: dict, order=0, sol_required=False):
    y = hmc_inv_pde['y']
    observe_mat = hmc_inv_pde['observe_mat']
    u, pu = solve_from_normal_input(inputs, inv_pde=hmc_inv_pde, derivative=1)
    s = observe_mat @ u
    ds = observe_mat @ pu
    sigma_y, sigma_theta = hmc_inv_pde['noise_ob'], hmc_inv_pde['sigma_theta']
    if order == 0:
        loglik = -((y - s) ** 2).sum() / (2 * sigma_y ** 2)
        logpri = - (inputs ** 2).sum() / (2 * sigma_theta ** 2)
        if sol_required:
            return -loglik - logpri, u, pu
        return -loglik - logpri
    elif order == 1:
        dloglik = (np.tile((y - s).reshape((-1, 1)), (1, hmc_inv_pde['kl_ndim'])) * ds).sum(0) / (sigma_y ** 2)
        dlogpri = - inputs / (sigma_theta ** 2)
        if sol_required:
            return -dloglik - dlogpri, u, pu
        return -dloglik - dlogpri


def hmc_evolve(hmc_inv_pde: dict, num_of_iter: int, state: str, start_theta, step_size: float = 0.16,
               num_of_leap_frog_steps: int = 10):
    acp_num = []
    timer = np.zeros(num_of_iter)
    current_theta = copy.copy(start_theta)
    current_potential = compute_potential(inputs=current_theta, hmc_inv_pde=hmc_inv_pde)
    if state == 'burnin':
        n = hmc_inv_pde['free_node'].size + hmc_inv_pde['fixed_node'].size
        x_data = np.zeros((hmc_inv_pde['kl_ndim'], 0))
        sol_data = np.zeros((n, 0))
        sol_grad_data = np.zeros((n, hmc_inv_pde['kl_ndim'], 0))
        potential_data = np.zeros((1, 0))
    else:
        sampled_theta = np.zeros((hmc_inv_pde['kl_ndim'], num_of_iter))

    for i in range(num_of_iter):
        time_start = time.clock()
        # initial momentum and theta for Hamiltonian dynamics
        proposed_momentum = np.random.randn(hmc_inv_pde['kl_ndim'])
        current_momentum = copy.copy(proposed_momentum)
        proposed_theta = copy.copy(current_theta)

        # Hamiltonian dynamics with leap frog
        random_num_leap_forg = np.random.choice(num_of_leap_frog_steps) + 1
        for _ in range(random_num_leap_forg):
            proposed_momentum -= step_size / 2 * compute_potential(inputs=proposed_theta, hmc_inv_pde=hmc_inv_pde, order=1)
            proposed_theta += step_size * proposed_momentum
            proposed_momentum -= step_size / 2 * compute_potential(inputs=proposed_theta, hmc_inv_pde=hmc_inv_pde, order=1)
        proposed_momentum = - proposed_momentum

        # compute potential
        if state == 'burnin':
            proposed_potential, u, pu = compute_potential(inputs=proposed_theta, hmc_inv_pde=hmc_inv_pde, sol_required=True)
        else:
            proposed_potential = compute_potential(inputs=proposed_theta, hmc_inv_pde=hmc_inv_pde)

        # compute Hamiltonian function
        current_H = current_potential + 0.5 * (current_momentum ** 2).sum()
        proposed_H = proposed_potential + 0.5 * (proposed_momentum ** 2).sum()
        ratio = current_H - proposed_H

        # accept or reject
        if math.isfinite(ratio) and (ratio > np.log(np.random.uniform())):
            current_theta = proposed_theta
            current_potential = proposed_potential
            acp_num.append(1)
            if state == 'burnin':
                x_data = np.concatenate((x_data, proposed_theta.reshape((-1, 1))), axis=1)
                sol_data = np.concatenate((sol_data, u.reshape((-1, 1))), axis=1)
                sol_grad_data = np.concatenate((sol_grad_data, pu[:, :, np.newaxis]), axis=2)
                potential_data = np.concatenate((potential_data, np.array([[proposed_potential]])), axis=1)
        else:
            acp_num.append(0)
        timer[i] = time.clock() - time_start

        # saved sampled theta
        if state != 'burnin':
            sampled_theta[:, i] = current_theta

        # print current progress
        if (i + 1) % 100 == 0:
            print('finished %d iterations.' % (i + 1))
            print('acceptance rate of latest 100 iterations %.2f' % (sum(acp_num[-100:]) / 100))

    if state == 'burnin':
        return current_theta, acp_num, timer, x_data, sol_data, sol_grad_data, potential_data
    else:
        return sampled_theta, acp_num, timer


