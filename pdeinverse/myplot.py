import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List


def load_hmc_npz_data(path: str):
    f = np.load(path)
    sampled_theta = f['sampled_theta']
    acp_num = f['acp_num']
    timer = f['timer']
    f.close()
    return sampled_theta, acp_num, timer


def search_file_name(prefix: str, key_word: str, excluded: List[str]):
    # read all files in a directory
    hmc_files, rns_files, dd_files = [], [], []
    for i, j, k in os.walk(prefix):
        for file in k:
            if file.endswith(key_word) and file not in excluded:
                if file.startswith('hmc_data'):
                    hmc_files.append(file)
                elif file.startswith('hmc_rns'):
                    rns_files.append(file)
                elif file.startswith('hmc_dd'):
                    dd_files.append(file)
    return hmc_files, rns_files, dd_files


def plot_posterior_statistics(index: int, figsize: tuple, theta: np.ndarray, kl_modes: np.ndarray, points: np.ndarray,
                              stat: str, barlim: tuple):
    post_sample = theta
    if stat == 'mean':
        field_func = np.mean(kl_modes @ post_sample, axis=1)
    elif stat == 'var':
        field_func = np.var(kl_modes @ post_sample, axis=1)
    elif stat == 'std':
        field_func = np.std(kl_modes @ post_sample, axis=1)
    x, y = points[:, 0], points[:, 1]

    plt.figure(index, figsize=figsize)
    plt.tricontourf(x, y, field_func, np.arange(barlim[0], barlim[1], barlim[2]))
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    return


def plot_1d_distribution(axis, x_data, xlim):
    sns.kdeplot(x_data, ax=axis)
    xlim = axis.set_xlim(xlim)
    return xlim


def plot_2d_distribution(axis, df, xlim, ylim):
    sns.kdeplot(data=df, ax=axis, cmap="coolwarm")
    axis.xaxis.label.set_visible(False)
    axis.yaxis.label.set_visible(False)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    return


def plot_marginal_joint_distributions(theta: np.ndarray, indices: List[int], lower_bound: np.ndarray,
                                      upper_bound: np.ndarray):
    fig, axs = plt.subplots(len(indices), len(indices), sharex=False, sharey=False, figsize=(10, 10))
    for i1, i in enumerate(indices):
        for j1, j in enumerate(indices):
            if i > j:
                axs[j1, i1].remove()
            elif i == j:
                plot_1d_distribution(axis=axs[j1, i1], x_data=theta[i, :], xlim=[lower_bound[i], upper_bound[i]])
                axs[j1, i1].set_yticks([])
                axs[j1, i1].set_xticks([])
            else:
                df = pd.DataFrame(theta[[i, j], :].transpose(), columns={'x', 'y'})
                plot_2d_distribution(axis=axs[j1, i1], df=df, xlim=[lower_bound[i], upper_bound[i]],
                                     ylim=[lower_bound[j], upper_bound[j]])
                if i != indices[0]:
                    axs[j1, i1].set_yticks([])
                if j != indices[-1]:
                    axs[j1, i1].set_xticks([])
    return
