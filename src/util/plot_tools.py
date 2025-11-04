import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R
import random

from . import quat_tools


# font = {'family' : 'Times New Roman',
#          'size'   : 18
#          }
# mpl.rc('font', **font)


def plot_ds(q_train, q_test, q_att):
    
    q_train_att = quat_tools.riem_log(q_att, q_train)
    q_test_att  = quat_tools.riem_log(q_att, q_test)

    q_train_3d = quat_tools.map(q_att, q_train_att)
    q_test_3d = quat_tools.map(q_att, q_test_att)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(q_train_3d[:, 0], q_train_3d[:, 1], q_train_3d[:, 2], 'o', color='k', s=1, alpha=0.4, label="Demonstration")
    ax.plot(q_test_3d[:, 0], q_test_3d[:, 1], q_test_3d[:, 2], color= 'b', linewidth=3)

    ax.set_xlabel(r'$\xi_1$', fontsize=38, labelpad=20)
    ax.set_ylabel(r'$\xi_2$', fontsize=38, labelpad=20)
    ax.set_zlabel(r'$\xi_3$', fontsize=38, labelpad=20)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis='z', which='major', pad=15)
    ax.axis('equal')
    ax.view_init(elev=30, azim=-20)




def plot_traj(q_train, q_test=None):
    """ q_list: nested list of quaternion objects"""

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()
    colors = ['lightsalmon', 'red', 'peru', 'maroon']
    for l in range(len(q_train)):
        q_l = np.array([q.as_quat() for q in q_train[l]])
        for k in range(4):
            ax.plot(q_l[:, k], linewidth=1, color=colors[k], alpha=0.4)

    if q_test is not None:
        q_l = np.array([q.as_quat() for q in q_test])
        for k in range(4):
            ax.plot(q_l[:, k], linewidth=3, color=colors[k], alpha=1)






def plot_omega(omega_test):

    # omega_test = np.vstack(omega_test)
    M, N = omega_test.shape
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(3):
        axs[k].scatter(np.arange(M), omega_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])



def plot_gamma(gamma_arr, **argv):

    M, K = gamma_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(M), gamma_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])
    else:
        axs[0].set_title(r"$\gamma(\cdot)$ over Time")



def plot_gmm(p_in, gmm):
    
    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    for k in range(K):
        label_k =np.where(label == k)[0]
        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)

        ax.text(loc[0], loc[1], loc[2]+0.03, str(k+1), fontsize=20)

        r = gmm.gaussian_list[k]["mu"]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)



    ax.axis('equal')
    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


