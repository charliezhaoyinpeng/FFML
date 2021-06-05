import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter


def read_data(data_path):
    loss, dbc, acc, dp, eop, disc, cons = [], [], [], [], [], [], []
    with open(data_path, 'rb') as f:
        out = pickle.load(f)
    for i in range(len(out)):
        loss.append(out[i][0])
        dbc.append(out[i][1])
        acc.append(out[i][2])
        dp.append(out[i][3])
        eop.append(out[i][4])
        disc.append(out[i][5])
        cons.append(out[i][6])
    return loss, dbc, acc, dp, eop, disc, cons


def mean_std(data_path1, data_path2, data_path3):
    mean_loss, mean_dbc, mean_acc, mean_dp, mean_eop, mean_disc, mean_cons = [], [], [], [], [], [], []
    std_loss, std_dbc, std_acc, std_dp, std_eop, std_disc, std_cons = [], [], [], [], [], [], []
    loss1, dbc1, acc1, dp1, eop1, disc1, cons1 = read_data(data_path1)
    loss2, dbc2, acc2, dp2, eop2, disc2, cons2 = read_data(data_path2)
    loss3, dbc3, acc3, dp3, eop3, disc3, cons3 = read_data(data_path3)
    for i in range(len(loss1)):
        temp_losses = [loss1[i], loss2[i], loss3[i]]
        mean_loss.append(np.mean(temp_losses))
        std_loss.append(np.std(temp_losses))

        temp_dbc = [dbc1[i], dbc2[i], dbc3[i]]
        mean_dbc.append(np.mean(temp_dbc))
        std_dbc.append(np.std(temp_dbc))

        temp_accs = [acc1[i], acc2[i], acc3[i]]
        mean_acc.append(np.mean(temp_accs))
        std_acc.append(np.std(temp_accs))

        temp_dps = [dp1[i], dp2[i], dp3[i]]
        mean_dp.append(np.mean(temp_dps))
        std_dp.append(np.std(temp_dps))

        temp_eops = [eop1[i], eop2[i], eop3[i]]
        mean_eop.append(np.mean(temp_eops))
        std_eop.append(np.std(temp_eops))

        temp_discs = [disc1[i], disc2[i], disc3[i]]
        mean_disc.append(np.mean(temp_discs))
        std_disc.append(np.std(temp_discs))

        temp_conss = [cons1[i], cons2[i], cons3[i]]
        mean_cons.append(np.mean(temp_conss))
        std_cons.append(np.std(temp_conss))

    return mean_loss, mean_dbc, mean_acc, mean_dp, mean_eop, mean_disc, mean_cons, std_loss, std_dbc, std_acc, std_dp, std_eop, std_disc, std_cons


def chunk(index, n):
    step = index / (n)
    ans = [round(i * step) for i in range(1, n)]
    ans.append(index)
    ans.insert(0, 0)
    ans.insert(0, 0)
    print(ans)
    return ans


#############################################################################################################
# def get_md_ours(l, n):
#     res = sorted(l, reverse=True)
#     return res[:n]
#     # res = []
#     # for i in range(len(l)):
#     #     item = np.round(l[i], 4)
#     #     if item > 0 and len(res) < n and item < 0.01:
#     #         res.append(l[i])
#     # return res

# def get_up(l, n):
#     res = []
#     med = np.median(l)
#     max = np.max(l)
#     for i in range(len(l)):
#         if l[i] > med and len(res) < n and l[i] != max:
#             res.append(l[i])
#     return res

def get(res, n):
    m = round(len(res) / n) + 1
    ans = [res[i:i + m] for i in range(0, len(res), m)]
    res = list(map(np.median, ans))
    return res


def get2(res, n):
    m = round(len(res) / n) + 1
    ans = [res[i:i + m] for i in range(0, len(res), m)]
    res = list(map(np.mean, ans))
    return res


def get_high(l, n):
    res = []
    med = np.median(l)
    for i in range(len(l)):
        if l[i] > med:
            res.append(l[i])
    m = round(len(res) / n)
    ans = [res[i:i + m] for i in range(0, len(res), m)]
    res = list(map(np.median, ans))
    return res


def get_low(l, n):
    res = []
    med = np.median(l)
    for i in range(len(l)):
        if l[i] <= med:
            res.append(l[i])
    m = round(len(res) / n) + 1
    ans = [res[i:i + m] for i in range(0, len(res), m)]
    res = list(map(np.median, ans))
    return res


def get_small_min(l, n):
    m = round(len(l) / n) + 1
    ans = [l[i:i + m] for i in range(0, len(l), m)]
    res = list(map(np.min, ans))
    return res


def get_small_middle(l, n):
    m = round(len(l) / n) + 1
    ans = [l[i:i + m] for i in range(0, len(l), m)]
    res = list(map(np.max, ans))
    return np.array(res) / 2


def get_small_max(l, n):
    m = round(len(l) / n) + 1
    ans = [l[i:i + m] for i in range(0, len(l), m)]
    res = list(map(np.mean, ans))
    return np.array(res)


#############################################################################################################

def plot_line(y, std, info):
    x = np.array(range(1, len(y) + 1))
    y = np.array(y)
    std = np.array(std)
    plt.plot(x, y, color=info[0], label=info[3], marker=info[2])
    plt.fill_between(x, y - std, y + std, alpha=0.2, facecolor=info[1])


def plot_ax_line(y, std, info, ax1, ax2):
    x = np.array(range(1, len(y) + 1))
    y = np.array(y)
    std = np.array(std)
    ax1.plot(x, y, color=info[0], label=info[3], marker=info[2])
    ax2.plot(x, y, color=info[0], label=info[3], marker=info[2])
    ax1.fill_between(x, y - std, y + std, alpha=0.2, facecolor=info[1])
    ax2.fill_between(x, y - std, y + std, alpha=0.2, facecolor=info[1])


def plot_ax_line_special(y, std, info, ax1, ax2):
    x = np.array(range(1, len(y) + 1))
    y = np.array(y)
    std = np.array(std)
    ax1.plot([], color=info[0], label=info[3], marker=info[2])
    ax2.plot(x, y, color=info[0], label=info[3], marker=info[2])
    ax1.fill_between(x, y - std, y + std, alpha=0.2, facecolor=info[1])
    ax2.fill_between(x, y - std, y + std, alpha=0.2, facecolor=info[1])


def plot_broken_axis(pts, stds, ylim1, ylim2, eval_name, num_tasks, dataname, up=False, hasLegend=True):
    if up:
        gs = GridSpec(2, 2, height_ratios=[2, 2])
    else:
        gs = GridSpec(2, 2, height_ratios=[4, 1])
    fig = plt.figure()
    ax = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    ax2 = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))
    info_sum = [['black', 'darkgray', 'o', 'Ours'],
                ['green', 'lawngreen', '^', 'm-FTML'],
                ['red', 'lightcoral', '*', 'TWP'],
                ['yellow', 'khaki', 'X', 'OGDLC'],
                ['cyan', 'skyblue', 'd', 'AdpOLC'],
                ['darkviolet', 'plum', 'v', 'GenOLC']]
    for i in range(len(info_sum)):
        if i not in [1, 3, 4, 5, 6, 7]:
            plot_ax_line(pts[i], stds[i], info_sum[i], ax, ax2)
        else:
            plot_ax_line_special(pts[i], stds[i], info_sum[i], ax, ax2)

    ax.set_ylim(ylim1)  # outliers only
    ax2.set_ylim(ylim2)  # most of the data

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    if up:
        if hasLegend:
            ax2.legend(loc='best', fontsize='large')
        ax2.set_ylabel(eval_name, fontsize=15, fontweight='bold')
    else:
        if hasLegend:
            ax.legend(loc='best', fontsize='large')
        ax.set_ylabel(eval_name, fontsize=15, fontweight='bold')
    ax2.set_xlabel("Task Index", fontsize=11, fontweight='bold')

    # ax2.set_xticklabels(chunk(num_tasks, 7))
    ax2.tick_params(axis="x", labelsize=11)

    ax.tick_params(axis="y", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    ax.set_title("%s " % (dataname))
    # fig.savefig('%s-%s.eps' % (dataname, eval_name), format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    fig.savefig('../%s-%s.pdf' % (dataname, eval_name), dpi=600, bbox_inches='tight')


def plotting(pts, stds, ylim, eval_name, num_tasks, dataname, hasLegend=True):
    info_sum = [['black', 'darkgray', 'o', 'Ours'],
                ['green', 'lawngreen', '^', 'm-FTML'],
                ['red', 'lightcoral', '*', 'TWP'],
                ['yellow', 'khaki', 'X', 'OGDLC'],
                ['cyan', 'skyblue', 'd', 'AdpOLC'],
                ['darkviolet', 'plum', 'v', 'GenOLC']
        # ,['darkorange', 'bisque', 'p', 'PDFM-sw']
                ]
    fig, ax = plt.subplots()
    for i in range(len(info_sum)):
        plot_line(pts[i], stds[i], info_sum[i])
    plt.gca().set_ylim(ylim)

    if hasLegend:
        plt.legend(loc='best', fontsize='large')

    # ax.set_xticklabels(chunk(num_tasks, 5))

    # x_formatter = FixedFormatter(xlabels)
    # # x_locator = FixedLocator([1, 3, 5, 7, 9, 11, 13, 15])
    # x_locator = FixedLocator(list(range(1, 35)[::5]))
    # ax.xaxis.set_major_formatter(x_formatter)
    # ax.xaxis.set_major_locator(x_locator)
    plt.xlabel("Task Index", fontsize=11, fontweight='bold')
    ax.tick_params(axis="x", labelsize=11)

    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    ax.tick_params(axis="y", labelsize=11)

    plt.title("%s " % (dataname))
    # fig.savefig('%s-%s.eps' % (dataname, eval_name), format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    fig.savefig('../%s-%s.pdf' % (dataname, eval_name), dpi=600, bbox_inches='tight')


def plotting_eff(pts, stds, ylim, eval_name, num_tasks, dataname, xlabels, ylabels, hasLegend=True):
    info_sum = [['black', 'darkgray', 'o', 'Ours'],
                ['green', 'lawngreen', '^', 'm-FTML'],
                ['red', 'lightcoral', '*', 'TWP'],
                ['yellow', 'khaki', 'X', 'OGDLC'],
                ['cyan', 'skyblue', 'd', 'AdpOLC'],
                ['darkviolet', 'plum', 'v', 'GenOLC']]
    fig, ax = plt.subplots()
    for i in range(len(info_sum)):
        plot_line(pts[i], stds[i], info_sum[i])
    plt.gca().set_ylim(ylim)

    if hasLegend:
        plt.legend(loc='best', fontsize='large')

    x_formatter = FixedFormatter(xlabels)
    x_locator = FixedLocator([1, 3, 5, 7, 9, 11, 13, 15])
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    plt.xlabel("Task Index", fontsize=15, fontweight='bold')  # fontsize=11
    ax.tick_params(axis="x", labelsize=11)

    ax.set_yticklabels(ylabels)
    plt.ylabel(eval_name, fontsize=20, fontweight='bold')  # fontsize=15
    ax.tick_params(axis="y", labelsize=11)

    plt.title("%s " % (dataname))
    # fig.savefig('%s-%s.eps' % (dataname, eval_name), format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    fig.savefig('../%s-%s.pdf' % (dataname, "Efficency"), dpi=600, bbox_inches='tight')


def plotting_batches(pts, stds, ylim, eval_name, num_tasks, dataname, xlabels, ylabels, hasLegend=True):
    info_sum = [['black', 'darkgray', 'o', '|U| = 32'],
                ['green', 'lawngreen', '^', '|U| = 16'],
                ['red', 'lightcoral', '*', '|U| = 8'],
                ['cyan', 'skyblue', 'd', '|U| = t']]
    fig, ax = plt.subplots()
    for i in range(len(info_sum)):
        plot_line(pts[i], stds[i], info_sum[i])
    plt.gca().set_ylim(ylim)

    if hasLegend:
        plt.legend(loc='best', fontsize='large')

    x_formatter = FixedFormatter(xlabels)
    x_locator = FixedLocator([1, 3, 5, 7, 9, 11, 13, 15])
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    plt.xlabel("Task Index", fontsize=11, fontweight='bold')
    ax.tick_params(axis="x", labelsize=11)

    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    ax.tick_params(axis="y", labelsize=11)

    plt.title("%s " % (dataname))
    # fig.savefig('%s-%s.eps' % (dataname, eval_name), format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    fig.savefig('../%s-%s.pdf' % (dataname, eval_name), dpi=600, bbox_inches='tight')


def plotting_as(pts, stds, ylim, eval_name, num_tasks, dataname, hasLegend=True):
    info_sum = [['black', 'darkgray', 'o', 'full ours'],
                ['green', 'lawngreen', '^', 'w/o inner FC'],
                ['red', 'lightcoral', '*', 'w/o aug'],
                ['cyan', 'skyblue', 'd', 'w/o aug + outer FC']]
    fig, ax = plt.subplots()
    for i in range(len(info_sum)):
        plot_line(pts[i], stds[i], info_sum[i])
    plt.gca().set_ylim(ylim)

    if hasLegend:
        plt.legend(loc='best', fontsize='large')

    plt.xlabel("Task Index", fontsize=11, fontweight='bold')
    ax.tick_params(axis="x", labelsize=11)

    plt.ylabel(eval_name, fontsize=15, fontweight='bold')
    ax.tick_params(axis="y", labelsize=11)

    plt.title("%s " % (dataname))
    # fig.savefig('%s-%s.eps' % (dataname, eval_name), format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    fig.savefig('../%s-%s.pdf' % (dataname, eval_name + "-AS"), dpi=600, bbox_inches='tight')
