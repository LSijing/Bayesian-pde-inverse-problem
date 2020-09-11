# -*- coding: utf-8 -*-

import numpy as np
from numpy.core.multiarray import ndarray
from scipy.sparse import coo_matrix


def compute_uniform_triangulation_vmatlab(n=30, domain=(0, 0, 1, 1)):
    """
    Mimic the uniform 2d Delaunay triangulation in Matlab.

    Input:
        domain  :  rectangle [domain[0],domain[0]+domain[2]] * [domain[1],domain[1]+domain[3]]
            domain = [left bottom width height]
        n : 2-tuple, or integer : the number of intervals on x,y direction
            ( (n[0]+1)^(n[1]+1), or (n+1)^2 nodes including boundary,
            2*n[0]*n[1], or 2n^2 triangles )

    Output:
        points, tris, edges : matrices in triangulation
        x_mesh,y_mesh   : 2d meshgrid
    """

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
    tris = np.array(
        [[i + (j + 1) * (nx + 1), i + j * (nx + 1), (i + 1) + j * (nx + 1)] for i in range(nx) for j in range(ny)] +
        [[i + 1 + (j + 1) * (nx + 1), i + (j + 1) * (nx + 1), (i + 1) + j * (nx + 1)] for i in range(nx) for j in
         range(ny)])
    edges = np.array([[i, i + 1] for i in range(nx)] +
                     [[nx + j * (nx + 1), nx + (j + 1) * (nx + 1)] for j in range(ny)] +
                     [[i + 1 + ny * (nx + 1), i + ny * (nx + 1)] for i in range(nx)] +
                     [[(j + 1) * (nx + 1), j * (nx + 1)] for j in range(ny)])

    return points, tris, edges, X, Y


def compute_mass_matrix(tris, points):
    """
    Construct the mass matrix of P1 approximation on triangle mesh in a sparse way
        M_ij = \integral (Phi_i*Phi_j)
        where phi_i, i=1,2,...np are standard hat functions

    Output:
        mass_matrix : sparse mass matrix in COOrdinate format.
    """
    n_dof = points.shape[0]
    nt = tris.shape[0]

    row_idx: ndarray = np.tile(tris, (1, 3)).flatten()
    col_idx: ndarray = np.tile(tris.flatten(), (3, 1)).transpose().flatten()

    x1_elem = points[tris[:, 0], 0:1]
    x2_elem = points[tris[:, 1], 0:1]
    x3_elem = points[tris[:, 2], 0:1]
    y1_elem = points[tris[:, 0], -1:]
    y2_elem = points[tris[:, 1], -1:]
    y3_elem = points[tris[:, 2], -1:]

    area = 0.5 * np.abs((x2_elem - x1_elem) * (y3_elem - y1_elem) - (x3_elem - x1_elem) * (y2_elem - y1_elem))

    mass_loc = np.tile(np.array([2, 1, 1, 1, 2, 1, 1, 1, 2]), (nt, 1))

    value = mass_loc * np.tile(area, (1, 9)) / 12
    value = value.flatten()
    # print(value.shape, row_idx.shape, col_idx.shape)
    mass_matrix = coo_matrix((value, (row_idx, col_idx)), shape=(n_dof, n_dof))
    # print('PLinApprox : finished constructing mass matrix.')
    return mass_matrix


def get_int_bd_idx(tris, edges):
    """
    Return the indices of nodes in interior (iin) and on boundary (ibd).
    """
    ibd = np.unique(edges.flatten())
    iin = np.delete(np.unique(tris.flatten()), ibd)
    return iin, ibd


def compute_load_vector(tris, points, f, quad='midpoint'):
    """
    Construct the load vector of P1 approximation on triangle mesh in a sparse way.
        b_i = \integral ( phi_i * f ),
        where phi_i, i=1,2,...np are standard hat functions

    Input:
        f : 2d function
        quad : quadrature formula :
            'corner' : based on nodes
            'center' : based on center
            'midpoint' : based on mid points of three edges

    Output:
        b : sparse vector in COOrdinate format.
    """

    nt = tris.shape[0]
    row_idx_loadv = tris.flatten()
    col_idx_loadv = np.zeros(nt * 3)

    x1_elem = points[tris[:, 0], 0:1]
    x2_elem = points[tris[:, 1], 0:1]
    x3_elem = points[tris[:, 2], 0:1]
    y1_elem = points[tris[:, 0], -1:]
    y2_elem = points[tris[:, 1], -1:]
    y3_elem = points[tris[:, 2], -1:]

    area = 0.5 * np.abs((x2_elem - x1_elem) * (y3_elem - y1_elem) - (x3_elem - x1_elem) * (y2_elem - y1_elem))
    if quad == 'center':
        xc_elem = (x1_elem + x2_elem + x3_elem) / 3
        yc_elem = (y1_elem + y2_elem + y3_elem) / 3
        fbar = np.array([[f(xc_elem[i, 0], yc_elem[i, 0])] for i in range(nt)])
        bK = fbar * area / 3
        value_loadv = (np.tile(bK, (1, 3))).flatten()
    if quad == 'corner':
        bK = np.array(
            [[f(x1_elem[i, 0], y1_elem[i, 0]), f(x2_elem[i, 0], y2_elem[i, 0]), f(x3_elem[i, 0], y3_elem[i, 0])] for
             i in range(nt)])
        value_loadv = (bK * np.tile(area, (1, 3)) / 3).flatten()
    if quad == 'midpoint':
        x_ec = np.concatenate(((x1_elem + x2_elem) / 2, (x2_elem + x3_elem) / 2, (x3_elem + x1_elem) / 2), axis=1)
        y_ec = np.concatenate(((y1_elem + y2_elem) / 2, (y2_elem + y3_elem) / 2, (y3_elem + y1_elem) / 2), axis=1)
        bK = np.array([[f(x_ec[i, 0], y_ec[i, 0]) / 2 + f(x_ec[i, 2], y_ec[i, 2]) / 2,
                        f(x_ec[i, 0], y_ec[i, 0]) / 2 + f(x_ec[i, 1], y_ec[i, 1]) / 2,
                        f(x_ec[i, 1], y_ec[i, 1]) / 2 + f(x_ec[i, 2], y_ec[i, 2]) / 2] for i in range(nt)])
        value_loadv = (bK * np.tile(area, (1, 3)) / 3).flatten()

    b = coo_matrix((value_loadv, (row_idx_loadv, col_idx_loadv)), shape=(points.shape[0], 1))
    # print('PLinApprox : finished assembling load vector.')
    return b


def compute_stiffness_matrix(tris, points, coef_discrete_on_center):
    """
    Construct the stiffness matrix of P1-FEM in a sparse way.
        A_ij = \integral ( (\grad phi_i) * a * (\grad phi_j) ),
        where phi_i, i=1,2,...np are standard hat functions

    Input:
        coef_discrete_on_center : discrete (array) coef on centers of 'tris'

    Output:
        A : sparse matrix in COOrdinate format.

    """
    n = points.shape[0]

    row_idx = np.tile(tris, (1, 3)).flatten()
    col_idx = np.tile(tris.flatten(), (3, 1)).transpose().flatten()

    x1_elem = points[tris[:, 0], 0:1]
    x2_elem = points[tris[:, 1], 0:1]
    x3_elem = points[tris[:, 2], 0:1]
    y1_elem = points[tris[:, 0], -1:]
    y2_elem = points[tris[:, 1], -1:]
    y3_elem = points[tris[:, 2], -1:]

    area = 0.5 * np.abs((x2_elem - x1_elem) * (y3_elem - y1_elem) - (x3_elem - x1_elem) * (y2_elem - y1_elem))
    # averaged a at centers of triangles
    abar = coef_discrete_on_center.reshape((-1, 1))
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

    value_stiff = AK * np.tile(area, (1, 9)) * np.tile(abar, (1, 9))
    value_stiff = value_stiff.flatten()
    stiff_mat = coo_matrix((value_stiff, (row_idx, col_idx)), shape=(n, n))
    return stiff_mat


def compute_pde_dictionary(n: int, domain=(0, 0, 1, 1)):
    """
    output:
        a dictionary with 'free_node', 'fixed_node', 'tris', 'points', 'center' representing the triangulation and boundary
    """
    func = lambda x, y: x * (y < 1e-6) + (1 - x) * (y > 1 - 1e-6) + 0.0
    points, tris, edges, x_mesh, y_mesh = compute_uniform_triangulation_vmatlab(n)
    fixed_node = np.where((points[:, 1] < 1e-6) | (points[:, 1] > 1 - 1e-6))[0]
    free_node = np.delete(np.arange((n + 1) ** 2), fixed_node)
    xc = (points[tris[:, 0], 0:1] + points[tris[:, 1], 0:1] + points[tris[:, 2], 0:1]) / 3
    yc = (points[tris[:, 0], 1:2] + points[tris[:, 1], 1:2] + points[tris[:, 2], 1:2]) / 3
    center = np.concatenate((xc, yc), axis=1)
    g_D = np.array([func(points[fixed_node[i], 0], points[fixed_node[i], 1]) for i in range(fixed_node.size)])
    pde = {'points': points, 'tris': tris, 'free_node': free_node, 'fixed_node': fixed_node, 'center': center, 'g_D': g_D}
    return pde
