import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.lines as mlines

ER_IL_COLORS = {
    "er_ace": '#43b7c2',
    "er": '#ef8f12',
}

SEL_STRATEGY_COLORS = {
    'random': '#006D77',
    'herding': '#83C5BE',
    'rm': '#EF3E36',
    'closest': '#FAD8D6'
}


def label_alg(alg_name):
    if alg_name == 'er_ace':
        return 'ER ACE'
    elif alg_name.lower() == 'er_ace_teal_one_time':
        return 'ER ACE+TEAL (One-Time)'
    elif alg_name.lower() in ['er ace_teal_log_iterative', 'er_ace_teal_log_iterative']:
        return 'ER ACE+TEAL'
    elif alg_name.lower() == 'er_ace_herding':
        return 'ER ACE+Herding'
    elif alg_name.lower() == 'er_ace_rm':
        return 'ER ACE+Uncertainty'
    elif alg_name.lower() == 'er_ace_centered':
        return 'ER ACE+Centered'
    elif alg_name == 'er':
        return 'ER'
    elif alg_name.lower() == 'er_teal_one_time':
        return 'ER+TEAL (One-Time)'
    elif alg_name.lower() == 'er_teal_log_iterative':
        return 'ER+TEAL'
    elif alg_name.lower() == 'er_herding':
        return 'ER+Herding'
    elif alg_name.lower() == 'er_rm':
        return 'ER+Uncertainty'
    elif alg_name.lower() == 'er_centered':
        return 'ER+Centered'
    elif alg_name.lower() == 'herding':
        return 'Herding'
    elif alg_name.lower() == 'rm':
        return 'Uncertainty'
    elif alg_name.lower() == 'closest':
        return 'Closest'
    else:
        return alg_name


def get_threshold_matrix(dataset_name, avg_matrix):
    """ For weighted accuracy """
    n_classes = 100 if dataset_name == 'cifar100' else None
    factor = 10 if dataset_name == 'cifar100' else None
    ratio = [(i * factor) / n_classes for i in range(1, avg_matrix.shape[1]+1)]
    # The addition is (x/n_classes)*(1/x) where x is the number of unseen classes. So at the last task it's 0.
    addition = np.zeros(avg_matrix.shape[1])
    addition[:-1] += 1/n_classes
    return avg_matrix * np.array(ratio) + addition


def get_avg_matrix(runs_acc_dict):
    avg_matrix = np.zeros((len(runs_acc_dict), len(runs_acc_dict[0][0])))
    for i, run in enumerate(runs_acc_dict):
        run_res = np.zeros((len(run), len(run[0])))
        for j in range(len(run)):
            run_res[j, j:] = run[j]
        div = np.arange(1, len(run[0]) + 1)
        run_avg = np.sum(run_res, axis=0) / div
        avg_matrix[i, :] = run_avg
    return avg_matrix


def get_algs_runs_acc_dict(dataset_name, buffer_size, alg_names, seed=None, inc_model=None,
                           sel_strategy_name=None):
    algs_runs_acc_dict = {}
    for alg_name in alg_names:
        alg_path = f'results/{dataset_name}/buffer_{buffer_size}/{alg_name}'
        if sel_strategy_name is not None:
            alg_path += f"/{sel_strategy_name}"
        if inc_model is not None:
            alg_path += f"/{inc_model}"
        if seed is not None:
            alg_path += f"/seed_{seed}"

        if not os.path.exists(alg_path):
            continue

        runs_acc_dict = []
        for folder in os.listdir(alg_path):
            res_file_name = 'results.pyth'
            res_path = f'{alg_path}/{folder}/{res_file_name}'

            if os.path.exists(res_path):
                print(f'Loading {res_path}')
                alg_result_dict = torch.load(res_path)
                runs_acc_dict.append(alg_result_dict[f'exp_acc_dict'])
            elif 'teal' in folder or 'herding' in folder or 'rm' in folder or 'centered' in folder:
                continue
            else:
                print(f'Probably an error in folder {alg_path}/{folder}')

        avg_matrix = get_avg_matrix(runs_acc_dict)
        if sel_strategy_name is not None:
            algs_runs_acc_dict[sel_strategy_name] = avg_matrix
        else:
            algs_runs_acc_dict[alg_name] = avg_matrix

    return algs_runs_acc_dict


def get_accuracies_stats(algs_runs_exp_acc_dict, acc_type='normal', dataset_name=None):
    acc_stats = {}

    for alg_name, runs_accs_np in algs_runs_exp_acc_dict.items():
        runs_accs_np *= 100
        if acc_type == 'normal':
            exps_stats = {'mean': np.mean(runs_accs_np, axis=0),
                          'se': stats.sem(runs_accs_np, axis=0)}
        else:
            threshold_matrix = get_threshold_matrix(dataset_name, runs_accs_np)
            exps_stats = {'mean': np.mean(threshold_matrix, axis=0),
                          'se': stats.sem(threshold_matrix, axis=0)}

        acc_stats[alg_name] = exps_stats
    return acc_stats


def plot_legend(legend_dict, with_dashed=False):
    fig, ax = plt.subplots(figsize=(20, 10))
    handles = [None for _ in range(len(legend_dict))]
    for i, (description, color) in enumerate(legend_dict.items()):
        if with_dashed and '+TEAL' in description:
            handle = mlines.Line2D([], [], color=color, linewidth=10, label=description, marker='o', linestyle=(0, (1.5, 2)), markersize=35)
        elif with_dashed and '+Herding' in description:
            handle = mlines.Line2D([], [], color=color, linewidth=10, label=description, marker='o', linestyle='-.', markersize=35)
        else:
            handle = mlines.Line2D([], [], color=color, linewidth=10, label=description, marker='o', linestyle='-', markersize=35)
        handles[i] = handle

    legend = ax.legend(handles=handles, loc='center', frameon=True, fontsize=40, ncol=1,
                       columnspacing=0.5, handlelength=2)
    frame = legend.get_frame()
    frame.set_linewidth(5)  # Set border width
    frame.set_edgecolor('black')  # Set border color
    ax.axis('off')  # Turn off the axis

    # Calculate legend dimensions
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    legend_width = legend_bbox.width
    legend_height = legend_bbox.height

    # Adjust figure dimensions based on legend dimensions
    fig_width = legend_width + 5
    fig_height = legend_height + 0.5
    fig.set_size_inches(fig_width, fig_height)

    # Print the shape of the legend image
    legend_image = fig.canvas.renderer.buffer_rgba()
    print("Legend image shape:", legend_image.shape)

    plt.show()


def plot_teal_comp_mult_alg(dataset_name, buffer_size, alg_names=None, teal_type='log_iterative',
                            tasks=10):
    if alg_names is None:
        print("No alg_names provided")
        return
    else:
        teal_alg_names = []
        for alg_name in alg_names:
            teal_alg_names.append(f"{alg_name}/teal_{teal_type}")

    algs_runs_exp_acc_dict = get_algs_runs_acc_dict(dataset_name, buffer_size,
                                                    alg_names=alg_names+teal_alg_names)
    acc_stats = get_accuracies_stats(algs_runs_exp_acc_dict, dataset_name=dataset_name)

    if dataset_name == 'cifar100':
        dataset_num_classes = 100
    elif dataset_name in ['tinyimg', 'cub200']:
        dataset_num_classes = 200
    elif dataset_name == 'cifar10':
        dataset_num_classes = 10
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print(f"#######################\nbuffer_size: {buffer_size}")
    x = np.arange(1, tasks + 1) * (dataset_num_classes // tasks)
    fig, ax = plt.subplots()
    for i, alg_name in enumerate(alg_names):
        print(alg_name)
        means, ses = acc_stats[alg_name]['mean'], acc_stats[alg_name]['se']
        label = f"{label_alg(alg_name)}"
        ax.errorbar(x, means, yerr=ses, marker='.', color=ER_IL_COLORS[alg_name],
                    linewidth=2, elinewidth=1, label=label)

        teal_alg_name = f"{alg_name}/teal_{teal_type}"
        teal_means, teal_ses = acc_stats[teal_alg_name]['mean'], acc_stats[teal_alg_name]['se']
        teal_label = f"{label_alg(teal_alg_name)}"

        ax.errorbar(x, teal_means, yerr=teal_ses, marker='.', color=ER_IL_COLORS[alg_name],
                    linewidth=2, elinewidth=1, linestyle='--', label=teal_label)

        ax.set_xticks(x)
        ax.tick_params(axis='both', which='major', labelsize=15)
        if tasks > 10:
            ax.tick_params(axis='x', rotation=90)

        ax.set_xlabel("Number of classes", fontsize=22)
        pos = ax.get_position()
        fig.text(pos.x0 - 0.1, 0.55, "Accuracy (%)",
                 fontsize=22, rotation='vertical', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=0.13, hspace=0)

    # Add legend
    legend_dict = {label_alg(k): v for k, v in ER_IL_COLORS.items() if k in alg_names}
    for k, v in ER_IL_COLORS.items():
        if k in alg_names:
            legend_dict[label_alg(f"{label_alg(k)}_teal_{teal_type}")] = v
    plot_legend(legend_dict, with_dashed=True)

    fig.show()


def plot_sel_strategy_comp(dataset_name, buffer_sizes, sel_strategy_names=None, seed=0,
                           inc_model='slim_resnet18', name='naive_replay', teal_type='log_iterative'):
    buffer_dict = {buffer: {} for buffer in buffer_sizes}
    for buffer in buffer_sizes:
        for sel_strategy in sel_strategy_names:
            if sel_strategy == 'teal':
                sel_strategy = f"{sel_strategy}_{teal_type}"
            algs_runs_exp_acc_dict = get_algs_runs_acc_dict(dataset_name, buffer, alg_names=[name], seed=seed,
                                                            inc_model=inc_model, sel_strategy_name=sel_strategy)
            acc_stats = get_accuracies_stats(algs_runs_exp_acc_dict, dataset_name=dataset_name)
            buffer_dict[buffer].update({sel_strategy: {'last_mean': acc_stats[sel_strategy]['mean'][-2],
                                        'last_se': acc_stats[sel_strategy]['se'][-2]}})
    names = list(buffer_dict[buffer_sizes[0]].keys())

    algs_buffer_stats = {}
    for i, name in enumerate(names):
        alg_buffer_means = [buffer_dict[buffer_size][name]['last_mean'] for buffer_size in buffer_sizes]
        alg_buffer_ses = [buffer_dict[buffer_size][name]['last_se'] for buffer_size in buffer_sizes]
        algs_buffer_stats[name] = {'mean': np.array(alg_buffer_means), 'se': np.array(alg_buffer_ses)}

    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = (1 / len(buffer_sizes))
    x = np.arange(len(buffer_sizes))
    ax.plot([-1, 4], [0, 0], color='black', linewidth=0.5, linestyle=(0, (5, 6)))

    teal_means = algs_buffer_stats[f'teal_{teal_type}']['mean']
    teal_ses = algs_buffer_stats[f'teal_{teal_type}']['se']
    handles, labels = [], []
    for i, name in enumerate(names):
        if 'teal' in name:
            continue
        means, ses = algs_buffer_stats[name]['mean'], algs_buffer_stats[name]['se']
        diff_means = teal_means - means
        diff_ses = np.sqrt(ses ** 2 + teal_ses ** 2)
        bars = ax.bar(x - i * width + 0.1,
                      diff_means,
                      yerr=diff_ses, alpha=0.8, ecolor='black',
                      capsize=5,
                      width=width, color=SEL_STRATEGY_COLORS[name],
                      label=f'{label_alg(name)}')
        handles.append(bars)
        labels.append(rf'TEAL accuracy(%) $-$ {label_alg(name)} accuracy(%)')

        print(f"{name}: {diff_means}")
    ax.set_xticks(x - 0.4)
    ax.set_xticklabels([f"{val}" for val in buffer_sizes])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("Buffer size", fontsize=22)
    pos = ax.get_position()
    fig.text(pos.x0 - 0.09, 0.52, "Accuracy Improvement",
             fontsize=20, rotation='vertical', va='center')
    fig.tight_layout()
    fig.subplots_adjust(left=0.14, hspace=0)
    fig.show()

    # Creating a new figure for the legend
    fig_legend = plt.figure(figsize=(15, 4))
    fig_legend.legend(handles, labels, loc='center', fontsize=40)
    fig_legend.show()
    return


if __name__ == '__main__':
    # Example:
    dataset_name = 'cifar100'
    buffer_size = 300
    alg_names = ['er']
    sel_strategy_names = ['herding', 'rm', 'teal']
    teal_type = 'log_iterative'
    tasks = 10

    # plot_sel_strategy_comp(dataset_name, [300], sel_strategy_names=sel_strategy_names, seed=0, inc_model='slim_resnet18')
    # plot_teal_comp_mult_alg(dataset_name, buffer_size, alg_names=alg_names, teal_type=teal_type, tasks=tasks)