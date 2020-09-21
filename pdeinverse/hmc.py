# -*- coding: utf-8 -*-

from pdeinverse import elliptic
import numpy as np
import copy
import time

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import scipy
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_inverse_pde_dictionary(pde: dict, var, corr_length, noise_ob, sigma_theta, kl_ndim):
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
        tmp = tmp + (np.tile(pk, (1, n)) - np.tile(pk.reshape((1, -1)), (n, 1))) ** 2
    cov = var ** 2 * np.exp(-tmp / (2 * corr_length ** 2))
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
        for _ in range(num_of_leap_frog_steps):
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


class HMC():
    def __init__(self, domain=(0, 0, 1, 1), n=2 ** 5, n_ob=2 ** 5,
                 noise_ob=0.05, corrvar=1.0, corrlength=0.2, sigmatheta=1.0, kl_ndim=20):
        self.domain = domain
        self.gD = lambda x, y: x * (y < 1e-6) + (1 - x) * (y > 1 - 1e-6) + 0.0
        self.points, self.tris, self.edges, self.x_mesh, self.y_mesh = elliptic.compute_uniform_triangulation_vmatlab(n)
        self.fixnode = np.where((self.points[:, 1] < 1e-6) | (self.points[:, 1] > 1 - 1e-6))[0]
        self.freenode = np.delete(np.arange((n + 1) ** 2), self.fixnode)
        self.set_ob_operator(n_ob)
        self.sigmay = noise_ob
        self.sigmatheta = sigmatheta
        print('HMC : finished initializing Bayesian inverse problem')
        """
        Gaussian process with zero mean, Gaussian kernel.
        cov(x1,x2) = corrvar^2 * exp{-||x1-x2||^2/(2*corrlength^2)}
        """
        (n, d) = self.points.shape
        tmp = np.zeros((n, n))
        for k in range(d):
            pk = self.points[:, k:k + 1]
            tmp = tmp + (np.tile(pk, (1, np)) - np.tile(pk.reshape((1, -1)), (n, 1))) ** 2
        cov = (corrvar ** 2) * (np.exp(-tmp / (2 * corrlength ** 2)))
        w, v = np.linalg.eigh(cov)
        w = w[::-1]
        v = v[:, ::-1]
        self.kl_modes = v[:, 0:kl_ndim]
        self.kl_weights = np.sqrt(np.absolute(w))[0:kl_ndim]

    def set_ob_operator(self, n):
        """
        """
        domain = self.domain
        if isinstance(n, tuple):
            nx = n[0]
            ny = n[1]
        elif isinstance(n, int):
            nx = n
            ny = n
        x = np.linspace(domain[0], domain[0] + domain[2], nx + 1)
        y = np.linspace(domain[1], domain[1] + domain[3], ny + 1)
        [X, Y] = np.meshgrid(x, y)
        points = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
        # only observe away from source, exclude points near the RHS source
        #        points = points[((points - np.array([0.3,0.3]))**2).sum(1)>0.01]
        #        points = points[((points - np.array([0.7,0.3]))**2).sum(1)>0.01]
        #        points = points[((points - np.array([0.3,0.7]))**2).sum(1)>0.01]
        #        points = points[((points - np.array([0.7,0.7]))**2).sum(1)>0.01]
        Jsparse = np.array([np.where(np.absolute(points[i,] - self.points).sum(1) < 1e-8)[0] for i in
                            range(points.shape[0])]).flatten()
        #        Jsparse = np.array([ np.where((points[i,] == self.p).all(1))[0] for i in range(points.shape[0]) ]).flatten()
        self.Ob = coo_matrix((np.ones(points.shape[0]), (np.arange(points.shape[0]), Jsparse)),
                             shape=(points.shape[0], self.points.shape[0]))
        print('HMC : finished setting observation operator')
        return

    def get_observation(self, u):
        return self.Ob @ u

    def get_noisy_observation(self, u, sigma):
        # add a noise vector with i.i.d. N(0,sigma^2)
        Nob = self.Ob.shape[0]
        y = self.get_observation(u) + np.random.normal(0.0, sigma, Nob)
        return y

    def get_kl_realization(self, input_uniform=None, input_normal=None):
        (n, k) = self.kl_modes.shape
        if input_uniform is not None:
            coefs = norm.ppf(input_uniform)
        elif input_normal is not None:
            coefs = input_normal
        else:
            coefs = norm.ppf(np.random.uniform(size=k))
        approxGP2 = ((coefs * self.kl_weights) * self.kl_modes).sum(1)
        return approxGP2

    def set_true_observation(self, inputs):
        n, k = self.kl_modes.shape
        inputs = inputs[0:k]
        u = self.solve_from_normal_input(inputs)
        self.sol = u
        self.y = self.get_noisy_observation(u, self.sigmay)

    def get_stiffness_matrix(self, coef_discrete):
        """
        Construct the stiffness matrix of P1-FEM in a sparse way.
            A_ij = \integral ( (\grad phi_i) * a * (\grad phi_j) ), 
            where phi_i, i=1,2,...np are standard hat functions
        
        Input:
            coef_discrete : discrete (array) coefficient
            
        Output:
            A : sparse matrix in COOrdinate format.
        
        """
        a = coef_discrete
        p = self.p
        t = self.t

        np = p.shape[0]
        nt = t.shape[0]

        Isparse_stiff = np.tile(t, (1, 3)).flatten()
        Jsparse_stiff = np.tile(t.flatten(), (3, 1)).transpose().flatten()

        x1_elem = p[t[:, 0], 0:1]
        x2_elem = p[t[:, 1], 0:1]
        x3_elem = p[t[:, 2], 0:1]
        y1_elem = p[t[:, 0], -1:]
        y2_elem = p[t[:, 1], -1:]
        y3_elem = p[t[:, 2], -1:]

        area = 0.5 * np.abs((x2_elem - x1_elem) * (y3_elem - y1_elem) - (x3_elem - x1_elem) * (y2_elem - y1_elem))
        # averaged a at centers of triangles
        abar = np.array([[(a[t[i, 0]] + a[t[i, 1]] + a[t[i, 2]]) / 3] for i in range(nt)])
        # gradients of hat (P1-FEM) functions
        b_elem = np.concatenate((y2_elem - y3_elem, y3_elem - y1_elem, y1_elem - y2_elem), axis=1) / np.tile(area,
                                                                                                             (1, 3)) / 2
        c_elem = np.concatenate((x3_elem - x2_elem, x1_elem - x3_elem, x2_elem - x1_elem), axis=1) / np.tile(area,
                                                                                                             (1, 3)) / 2
        AK = np.concatenate((b_elem[:, 0:1] * b_elem[:, 0:1] + c_elem[:, 0:1] * c_elem[:, 0:1],
                             b_elem[:, 1:2] * b_elem[:, 0:1] + c_elem[:, 1:2] * c_elem[:, 0:1],
                             b_elem[:, 2:3] * b_elem[:, 0:1] + c_elem[:, 2:3] * c_elem[:, 0:1],
                             b_elem[:, 0:1] * b_elem[:, 1:2] + c_elem[:, 0:1] * c_elem[:, 1:2],
                             b_elem[:, 1:2] * b_elem[:, 1:2] + c_elem[:, 1:2] * c_elem[:, 1:2],
                             b_elem[:, 2:3] * b_elem[:, 1:2] + c_elem[:, 2:3] * c_elem[:, 1:2],
                             b_elem[:, 0:1] * b_elem[:, 2:3] + c_elem[:, 0:1] * c_elem[:, 2:3],
                             b_elem[:, 1:2] * b_elem[:, 2:3] + c_elem[:, 1:2] * c_elem[:, 2:3],
                             b_elem[:, 2:3] * b_elem[:, 2:3] + c_elem[:, 2:3] * c_elem[:, 2:3]), axis=1)

        Asparse_stiff = AK * np.tile(area, (1, 9)) * np.tile(abar, (1, 9))
        Asparse_stiff = Asparse_stiff.flatten()
        A = coo_matrix((Asparse_stiff, (Isparse_stiff, Jsparse_stiff)), shape=(np, np))
        # print('Inv_Ellip_2d : finished assembling stiffness matrix of discrete log(a)')
        return A

    def solve_from_normal_input(self, inputs, derivative=0):
        c = np.exp(self.get_kl_realization(input_normal=inputs))
        A = self.get_stiffness_matrix(c)
        A11 = A.tocsr()[self.freenode, :][:, self.freenode]
        n = self.freenode.size + self.fixnode.size
        u = np.zeros(n)
        u[self.fixnode] = np.array(
            [self.gD(self.p[self.fixnode[i], 0], self.p[self.fixnode[i], 1]) for i in range(self.fixnode.size)])
        u[self.freenode] = spsolve(A11, (-A.tocsr() @ u)[self.freenode])
        if derivative == 0:
            return u
        if derivative == 1:
            pu = np.zeros((n, self.kl_modes.shape[1]))
            for i in range(self.kl_modes.shape[1]):
                pc = c * self.kl_weights[i] * self.kl_modes[:, i]
                pA = self.get_stiffness_matrix(pc)
                pbLag = -pA.tocsr() @ u
                #            pu[:,i] = spsolve(A.tocsr()+self.robin_mat.tocsr(), pbLag.reshape((-1,1)) + self.robin_vec)
                pu[self.freenode, i] = spsolve(A11, pbLag[self.freenode])
            return u, pu

    def get_potential(self, inputs, y=None, order=0):
        if y is None:
            y = self.y
        u, pu = self.solve_from_normal_input(inputs, derivative=1)
        s = self.get_observation(u)
        ds = self.get_observation(pu)
        if order == 0:
            loglik = -((y - s) ** 2).sum() / (2 * self.sigmay ** 2)
            logpri = - (inputs ** 2).sum() / (2 * self.sigmatheta ** 2)
            return -loglik - logpri
        elif order == 1:
            dloglik = (np.tile((y - s).reshape((-1, 1)), (1, self.kl_modes.shape[1])) * ds).sum(0) / (self.sigmay ** 2)
            dlogpri = - inputs / (self.sigmatheta ** 2)
            return -dloglik - dlogpri


class HMCv2(HMC):
    """
    define the discrete log-coefficient on the centers of triangle elements
    """

    def __init__(self, domain=(0, 0, 1, 1), n=2 ** 5, n_ob=2 ** 5,
                 noise_ob=0.05, corrvar=1.0, corrlength=0.2, sigmatheta=1.0, kl_ndim=20):
        self.domain = domain
        self.gD = lambda x, y: x * (y < 1e-6) + (1 - x) * (y > 1 - 1e-6) + 0.0
        self.set_uniform_triangulation_vmatlab(n)
        self.fixnode = np.where((self.p[:, 1] < 1e-6) | (self.p[:, 1] > 1 - 1e-6))[0]
        self.freenode = np.delete(np.arange((n + 1) ** 2), self.fixnode)
        self.set_ob_operator(n_ob)
        self.sigmay = noise_ob
        self.sigmatheta = sigmatheta
        print('HMC : finished initializing Bayesian inverse problem')
        """
        v2: discretized on centers of triangle elements
        Gaussian process with zero mean, Gaussian kernel.
        cov(x1,x2) = corrvar^2 * exp{-||x1-x2||^2/(2*corrlength^2)}
        """
        self.center = (self.p[self.t[:, 0], :] + self.p[self.t[:, 1], :] + self.p[self.t[:, 2], :]) / 3
        (nc, d) = self.center.shape
        tmp = np.zeros((nc, nc))
        for k in range(d):
            pk = self.center[:, k:k + 1]
            tmp = tmp + (np.tile(pk, (1, nc)) - np.tile(pk.reshape((1, -1)), (nc, 1))) ** 2
        cov = (corrvar ** 2) * (np.exp(-tmp / (2 * corrlength ** 2)))
        w, v = np.linalg.eigh(cov)
        w = w[::-1]
        v = v[:, ::-1]
        self.kl_modes = v[:, 0:kl_ndim]
        self.kl_weights = np.sqrt(np.absolute(w))[0:kl_ndim]

    def get_stiffness_matrix(self, coef_discrete_on_center):
        a = coef_discrete_on_center
        p = self.p
        t = self.t

        np = p.shape[0]

        Isparse_stiff = np.tile(t, (1, 3)).flatten()
        Jsparse_stiff = np.tile(t.flatten(), (3, 1)).transpose().flatten()

        x1_elem = p[t[:, 0], 0:1]
        x2_elem = p[t[:, 1], 0:1]
        x3_elem = p[t[:, 2], 0:1]
        y1_elem = p[t[:, 0], -1:]
        y2_elem = p[t[:, 1], -1:]
        y3_elem = p[t[:, 2], -1:]

        area = 0.5 * np.abs((x2_elem - x1_elem) * (y3_elem - y1_elem) - (x3_elem - x1_elem) * (y2_elem - y1_elem))
        # averaged a at centers of triangles
        abar = a.reshape((-1, 1))
        # gradients of hat (P1-FEM) functions
        b_elem = np.concatenate((y2_elem - y3_elem, y3_elem - y1_elem, y1_elem - y2_elem), axis=1) / np.tile(area,
                                                                                                             (1, 3)) / 2
        c_elem = np.concatenate((x3_elem - x2_elem, x1_elem - x3_elem, x2_elem - x1_elem), axis=1) / np.tile(area,
                                                                                                             (1, 3)) / 2
        AK = np.concatenate((b_elem[:, 0:1] * b_elem[:, 0:1] + c_elem[:, 0:1] * c_elem[:, 0:1],
                             b_elem[:, 1:2] * b_elem[:, 0:1] + c_elem[:, 1:2] * c_elem[:, 0:1],
                             b_elem[:, 2:3] * b_elem[:, 0:1] + c_elem[:, 2:3] * c_elem[:, 0:1],
                             b_elem[:, 0:1] * b_elem[:, 1:2] + c_elem[:, 0:1] * c_elem[:, 1:2],
                             b_elem[:, 1:2] * b_elem[:, 1:2] + c_elem[:, 1:2] * c_elem[:, 1:2],
                             b_elem[:, 2:3] * b_elem[:, 1:2] + c_elem[:, 2:3] * c_elem[:, 1:2],
                             b_elem[:, 0:1] * b_elem[:, 2:3] + c_elem[:, 0:1] * c_elem[:, 2:3],
                             b_elem[:, 1:2] * b_elem[:, 2:3] + c_elem[:, 1:2] * c_elem[:, 2:3],
                             b_elem[:, 2:3] * b_elem[:, 2:3] + c_elem[:, 2:3] * c_elem[:, 2:3]), axis=1)

        Asparse_stiff = AK * np.tile(area, (1, 9)) * np.tile(abar, (1, 9))
        Asparse_stiff = Asparse_stiff.flatten()
        A = coo_matrix((Asparse_stiff, (Isparse_stiff, Jsparse_stiff)), shape=(np, np))
        return A


class HMCNN(HMCv2):
    """
    neural network solver for forward problems
    """

    def get_potential(self, inputs, y=None, order=0, solrequired=False):
        if y is None:
            y = self.y
        u, pu = self.solve_from_normal_input(inputs, derivative=1)
        s = self.get_observation(u)
        ds = self.get_observation(pu)
        if order == 0:
            loglik = -((y - s) ** 2).sum() / (2 * self.sigmay ** 2)
            logpri = - (inputs ** 2).sum() / (2 * self.sigmatheta ** 2)
            if solrequired == False:
                return -loglik - logpri
            elif solrequired == True:
                return -loglik - logpri, u, pu
        elif order == 1:
            dloglik = (np.tile((y - s).reshape((-1, 1)), (1, self.kl_modes.shape[1])) * ds).sum(0) / (self.sigmay ** 2)
            dlogpri = - inputs / (self.sigmatheta ** 2)
            if solrequired == False:
                return -dloglik - dlogpri
            elif solrequired == True:
                return -dloglik - dlogpri, u, pu

    def set_data_driven_solver(self, phi_sol, phi_grad, net_sol, net_grad, ave_sol, std_sol, ave_grad, std_grad):
        self.phi_sol = phi_sol[:, 1:]
        self.mean_sol = phi_sol[:, 0]
        self.phi_grad = phi_grad[:, 1:]
        self.mean_grad = phi_grad[:, 0]
        self.net_sol = net_sol
        self.net_grad = net_grad
        self.ave_sol = ave_sol
        self.ave_grad = ave_grad
        self.std_sol = std_sol
        self.std_grad = std_grad

    def solve_through_net(self, inputs, derivative=0):
        x = torch.from_np(inputs)
        u = self.phi_sol @ (self.net_sol(x) * self.std_sol + self.ave_sol).detach().np() + self.mean_sol
        if derivative == 0:
            return u
        if derivative == 1:
            n = self.freenode.size + self.fixnode.size
            pu = np.zeros((n, self.kl_modes.shape[1]))
            for i in range(self.kl_modes.shape[1]):
                pu[:, i] = self.phi_grad @ (
                        self.net_grad[i](x) * self.std_grad[i] + self.ave_grad[i]).detach().np() + self.mean_grad
            return u, pu

    def get_potential_through_net(self, inputs, y=None, order=0, solrequired=False):
        if y is None:
            y = self.y
        u, pu = self.solve_through_net(inputs, derivative=1)
        s = self.get_observation(u)
        ds = self.get_observation(pu)
        if order == 0:
            loglik = -((y - s) ** 2).sum() / (2 * self.sigmay ** 2)
            logpri = - (inputs ** 2).sum() / (2 * self.sigmatheta ** 2)
            if solrequired == False:
                return -loglik - logpri
            elif solrequired == True:
                return -loglik - logpri, u, pu
        elif order == 1:
            dloglik = (np.tile((y - s).reshape((-1, 1)), (1, self.kl_modes.shape[1])) * ds).sum(0) / (self.sigmay ** 2)
            dlogpri = - inputs / (self.sigmatheta ** 2)
            if solrequired == False:
                return -dloglik - dlogpri
            elif solrequired == True:
                return -dloglik - dlogpri, u, pu


class HMCGalerkin(HMCv2):
    """
    Galerkin projection for forward problem
    """

    def get_potential(self, inputs, y=None, order=0, solrequired=False):
        if y is None:
            y = self.y
        u, pu = self.solve_from_normal_input(inputs, derivative=1)
        s = self.get_observation(u)
        ds = self.get_observation(pu)
        if order == 0:
            loglik = -((y - s) ** 2).sum() / (2 * self.sigmay ** 2)
            logpri = - (inputs ** 2).sum() / (2 * self.sigmatheta ** 2)
            if solrequired == False:
                return -loglik - logpri
            elif solrequired == True:
                return -loglik - logpri, u, pu
        elif order == 1:
            dloglik = (np.tile((y - s).reshape((-1, 1)), (1, self.kl_modes.shape[1])) * ds).sum(0) / (self.sigmay ** 2)
            dlogpri = - inputs / (self.sigmatheta ** 2)
            if solrequired == False:
                return -dloglik - dlogpri
            elif solrequired == True:
                return -dloglik - dlogpri, u, pu

    def set_galerkin_solver(self, phi_sol, phi_grad):
        self.phi_sol = phi_sol[:, 1:]
        self.mean_sol = phi_sol[:, 0]
        self.phi_grad = phi_grad[:, 1:]
        self.mean_grad = phi_grad[:, 0]

    def solve_through_galerkin(self, inputs, derivative=0):
        c = np.exp(self.get_kl_realization(input_normal=inputs))
        A = self.get_stiffness_matrix(c)
        A11 = A.tocsr()[self.freenode, :][:, self.freenode]
        n = self.freenode.size + self.fixnode.size
        u = np.zeros(n)

        #        A_reduced =  self.phi_sol[self.freenode,:].transpose() @ A11 @ self.phi_sol[self.freenode,:]
        #        u_reduced = self.mean_sol + self.phi_sol @ scipy.linalg.solve(A_reduced, self.phi_sol[self.freenode,:].transpose() @ (-A.tocsr() @ u)[self.freenode], assume_a = 'pos' )
        u[self.fixnode] = np.array(
            [self.gD(self.p[self.fixnode[i], 0], self.p[self.fixnode[i], 1]) for i in range(self.fixnode.size)])
        u[self.freenode] = spsolve(A11, (-A.tocsr() @ u)[self.freenode])
        if derivative == 0:
            return u
        if derivative == 1:
            pu_reduced = np.zeros((n, self.kl_modes.shape[1]))
            Agrad_reduced = self.phi_grad[self.freenode, :].transpose() @ A11 @ self.phi_grad[self.freenode, :]
            for i in range(self.kl_modes.shape[1]):
                pc = c * self.kl_weights[i] * self.kl_modes[:, i]
                pA = self.get_stiffness_matrix(pc)
                pbLag = -pA.tocsr() @ u
                pu_reduced[:, i] = self.mean_grad + self.phi_grad @ scipy.linalg.solve(Agrad_reduced,
                                                                                       self.phi_grad[self.freenode,
                                                                                       :].transpose() @ pbLag[
                                                                                           self.freenode])
            #            pu[:,i] = spsolve(A.tocsr()+self.robin_mat.tocsr(), pbLag.reshape((-1,1)) + self.robin_vec)
            #                pu[self.freenode,i] = spsolve(A11, pbLag[self.freenode])
            return u, pu_reduced

    def get_potential_through_galerkin(self, inputs, y=None, order=0, solrequired=False):
        if y is None:
            y = self.y
        u, pu = self.solve_through_galerkin(inputs, derivative=1)
        s = self.get_observation(u)
        ds = self.get_observation(pu)
        if order == 0:
            loglik = -((y - s) ** 2).sum() / (2 * self.sigmay ** 2)
            logpri = - (inputs ** 2).sum() / (2 * self.sigmatheta ** 2)
            if solrequired == False:
                return -loglik - logpri
            elif solrequired == True:
                return -loglik - logpri, u, pu
        elif order == 1:
            dloglik = (np.tile((y - s).reshape((-1, 1)), (1, self.kl_modes.shape[1])) * ds).sum(0) / (self.sigmay ** 2)
            dlogpri = - inputs / (self.sigmatheta ** 2)
            if solrequired == False:
                return -dloglik - dlogpri
            elif solrequired == True:
                return -dloglik - dlogpri, u, pu


class SolutionMap(nn.Module):
    def __init__(self, size_in=20, size_out=20, size_hidden=30):
        super(SolutionMap, self).__init__()
        self.size = size_hidden
        self.inp = nn.Linear(size_in, self.size)
        self.l1 = nn.Linear(self.size, self.size)
        self.l2 = nn.Linear(self.size, self.size)
        self.l3 = nn.Linear(self.size, self.size)
        # self.l4 = nn.Linear(self.size,self.size)
        # self.l5 = nn.Linear(self.size,self.size)
        self.out = nn.Linear(self.size, size_out)
        self.acti = F.relu

    def forward(self, x):
        y = self.inp(x)
        y = self.acti(self.l1(y)) + y
        y = self.acti(self.l2(y)) + y
        y = self.acti(self.l3(y)) + y
        # y = self.acti(self.l4(y))+y
        # y = self.acti(self.l5(y))+y
        y = self.out(y)
        return y


def trainit(net, x_train, y_train, opt='Adam', epochs=100, lr=0.02, num_iter=5):
    """
    rowwise np data
    """
    xdata = torch.from_numpy(x_train)
    ydata = torch.from_numpy(y_train)
    Ave = ydata.mean()
    Std = ydata.std()
    ydata = (ydata - Ave) / Std

    learning_rate = lr
    criterion = nn.MSELoss()
    training_time_array = np.zeros(0)
    training_loss_array = np.zeros(0)

    for i in range(num_iter):
        if opt == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate / (2 ** i))
        elif opt == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate / (2 ** i))
        print(i + 1, 'out of', num_iter)
        # Adam(net.parameters(),lr = learning_rate/(4**i))
        for j in range(epochs):
            x = xdata
            y = ydata
            yout = net(x)
            loss = criterion(yout, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_time_array = np.concatenate([training_time_array, np.array(i * epochs + j + 1).reshape(1)])
            training_loss_array = np.concatenate([training_loss_array,
                                                  ((net(xdata) * Std + Ave - (ydata * Std + Ave)) ** 2).mean(
                                                      1).mean().detach().np().reshape(1)], 0)
    #            if (j%(epochs//5)==epochs//5-1):
    #                print('loss ### %.6f ### %.6f'%(loss.item(),training_loss_array[-1]))

    return Ave, Std, training_time_array, training_loss_array


if __name__ == '__main__':
    np.random.seed(12)

    f = sio.loadmat('ePDE.mat')
    momentum = sio.loadmat('emomentum.mat')['momentum']

    N = 30
    Nob = 10
    num_kl = 20

    prob = HMCv2(n=N, n_ob=Nob, noise_ob=0.1, corrvar=1.0, corrlength=0.2, sigmatheta=0.5, kl_ndim=num_kl)
    prob.y = f['y'].reshape((Nob + 1, Nob + 1)).transpose().flatten()

    #    prob.set_true_observation(inputs=theta_true)

    #    plt.figure(5,figsize=(5,4))
    #    plt.pcolor(prob.x_mesh, prob.y_mesh, (prob.sol).reshape((N+1,N+1)), cmap='viridis')
    #    plt.axis([prob.domain[0], prob.domain[0]+prob.domain[2], prob.domain[1], prob.domain[1]+prob.domain[3]])
    #    plt.colorbar(aspect=12)
    #
    #    (u,pu) = prob.solve_from_normal_input(inputs=theta_true, derivative=1)
    #    plt.figure(7,figsize=(5,4))
    #    plt.pcolor(prob.x_mesh, prob.y_mesh, (pu[:,0]).reshape((N+1,N+1)), cmap='viridis')
    #    plt.axis([prob.domain[0], prob.domain[0]+prob.domain[2], prob.domain[1], prob.domain[1]+prob.domain[3]])
    #    plt.colorbar(aspect=12)

    Startpoint = np.zeros((num_kl))
    CurrentTheta = Startpoint
    CurrentU = prob.get_potential(CurrentTheta)

    Accepted, Proposed = 0, 0
    NumOfLeapFrogSteps = 10
    StepSize = 0.16
    IterNum = 5000
    BurnIn = 1000

    thetaSaved = np.zeros((IterNum - BurnIn, num_kl))

    for iiter in range(IterNum):
        if (iiter + 1) % 100 == 0:
            print('%d iterations completed.' % (iiter + 1))
            print(Accepted / Proposed)
            Accepted, Proposed = 0, 0

        ProposedMomentum = momentum[:, iiter]
        CurrentMomentum = ProposedMomentum

        Proposed = Proposed + 1
        ProposedTheta = CurrentTheta

        # Use random Leapfrog steps
        RandomLeapFrogSteps = np.random.choice(NumOfLeapFrogSteps) + 1
        # Perform leapfrog steps
        for StepNum in range(NumOfLeapFrogSteps):
            # HMC
            ProposedMomentum = ProposedMomentum - StepSize / 2 * prob.get_potential(ProposedTheta, order=1)
            ProposedTheta = ProposedTheta + StepSize * (ProposedMomentum)
            ProposedMomentum = ProposedMomentum - StepSize / 2 * prob.get_potential(ProposedTheta, order=1)

        ProposedMomentum = - ProposedMomentum
        # calculate potential
        ProposedU = prob.get_potential(ProposedTheta)

        # calculate Hamiltonian function
        CurrentH = CurrentU + 0.5 * (CurrentMomentum ** 2).sum()
        ProposedH = ProposedU + 0.5 * (ProposedMomentum ** 2).sum()

        # calculate the ratio
        Ratio = -ProposedH + CurrentH

        if math.isfinite(Ratio) and (Ratio > np.log(np.random.uniform())):
            CurrentTheta = ProposedTheta
            CurrentU = ProposedU
            Accepted = Accepted + 1

        # Save samples if required
        if iiter > BurnIn:
            thetaSaved[iiter - BurnIn, :] = CurrentTheta

        # Start timer after burn-in
        if iiter == BurnIn:
            print('Burn-in complete, now drawing samples.')

    #    plt.figure(1)
    #    plt.plot(((thetaSaved - theta_true) ** 2).sum(1))
    #    plt.figure(2)
    #    plt.plot((((prob.kl_modes @ thetaSaved.transpose()).transpose() - (prob.kl_modes @ theta_true).transpose()) **2 ).sum(1))
    for j in range(5):
        plt.figure(j + 10)
        plt.plot(thetaSaved[::2, j])
        print(thetaSaved[::2, j].mean(), thetaSaved[::2, j].std())
