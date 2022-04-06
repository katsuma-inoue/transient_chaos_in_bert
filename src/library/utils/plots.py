import matplotlib.pyplot as plt
import numpy as np
import os


def plot_chaos_length_map(
        chaos_lengths_map, mesh_num, pert_norm, save_dir, suffix=""):
    """Plot the chaos length map.
    """
    fig, ax = plt.subplots(figsize=(mesh_num*2.5+3, mesh_num*2.5+3))
    im = ax.imshow(chaos_lengths_map, cmap=plt.cm.magma)
    plt.colorbar(im)

    ax.set_xticks(np.arange(chaos_lengths_map.shape[1]))
    ax.set_yticks(np.arange(chaos_lengths_map.shape[0]))

    ticklabels = np.arange(
        -mesh_num*pert_norm, (mesh_num+1)*pert_norm, pert_norm)
    ticklabels = [f"{label:.3e}" for label in ticklabels]
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='right',
        rotation_mode='anchor'
    )
    plt.rcParams['font.size'] = 6
    for i in range(chaos_lengths_map.shape[0]):
        for j in range(chaos_lengths_map.shape[1]):
            ax.text(
                j, i, f"{chaos_lengths_map[i, j]:.3e}",
                ha='center', va='center', color='w', fontsize=8)
    ax.set_title("Lengths of Transient Chaos trajectory map.")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"perturbation_map{suffix}.png"))
    plt.close()


def plot_compare_trajectories(
        trajectory0, trajectory1, dir_path,
        show_dim_num=5, max_step=1e6, min_step=1e5):
    os.makedirs(dir_path, exist_ok=True)
    plt.figure(figsize=(15, 6*show_dim_num))
    plt.rcParams["font.size"] = 16
    for j in range(0, show_dim_num):
        show_dim = j
        plt.subplot(show_dim_num*2, 1, j+1)
        _plot_trajectory_dim(
            j, trajectory0, show_dim, show_dim_num, max_step, max_step, color='b')
        plt.subplot(show_dim_num*2, 1, j + show_dim_num + 1)
        _plot_trajectory_dim(
            j, trajectory1, show_dim, show_dim_num, max_step, min_step, color='r')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.0)
    save_path = os.path.join(dir_path, "compare_trajectories.png")
    plt.savefig(save_path)
    plt.close()


def plot_simple_trajectory(
        trajectory, dir_path,
        show_dim_num=5, max_step=1e6):
    os.makedirs(dir_path, exist_ok=True)
    plt.figure(figsize=(15, 4*show_dim_num))
    plt.rcParams["font.size"] = 16
    for j in range(0, show_dim_num):
        show_dim = j
        plt.subplot(show_dim_num, 1, j+1)
        _plot_trajectory_dim(
            j, trajectory, show_dim, show_dim_num, max_step, max_step, color='b')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.0)
    save_path = os.path.join(dir_path, "trajectories.png")
    plt.savefig(save_path)
    plt.close()


def _plot_trajectory_dim(
        j, trajectory, show_dim, show_dim_num, plot_max, plot_step, **kwargs):
    print(plot_step)
    show_seq = trajectory[j][:int(plot_step)]
    plt.plot(show_seq, **kwargs)
    plt.ylabel(f"dim={show_dim}")
    if j < show_dim_num*2-1:
        plt.tick_params(
            labelbottom=False,
            labeltop=False)
    plt.vlines(
            np.arange(0, trajectory.shape[-1], trajectory.shape[-1]//10),
            -3,
            3,
            'gray',
            linestyles='dashed', alpha=.5)
    plt.ylim([-3, 3])
    plt.xlim([0, plot_max])
    plt.hlines([0], 0, trajectory.shape[-1], 'gray', alpha=.5)
