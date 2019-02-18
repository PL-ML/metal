import os
from os.path import join as joinpath
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def parse_single_log(dirpath):

    reg = re.compile(r"'eval_result': (\d+)")
    cnt = 0
    total_cnt = 0

    for filename in os.listdir(dirpath):
        with open(joinpath(dirpath, filename), 'r') as f:
            total_cnt += 1
            m = reg.search(f.read())

            if m is None:
                continue

            print(filename)
            print(m.group(1))
            cnt += 1

    print('%i/%i' % (cnt, total_cnt))


def sort_spec(dirpath):
    reg = re.compile(r"'eval_result': (\d+)")

    solved_spec_ls = []
    unsolved_spec_ls = []

    for filename in os.listdir(dirpath):
        with open(joinpath(dirpath, filename), 'r') as f:

            m = reg.search(f.read())
            if m is not None:
                solved_spec_ls.append([filename, int(m.group(1))])
            else:
                unsolved_spec_ls.append([filename, 0])


    solved_spec_ls.sort(key=lambda x:x[1])

    print('mean', np.mean([e[1] for e in solved_spec_ls]))
    print('median', np.median([e[1] for e in solved_spec_ls]))

    # with open('../../benchmarks/test_nofalse', 'r') as ff:
    #     test_spec = [line.strip() for line in ff]
    #
    # rank_dict = dict(solved_spec_ls+unsolved_spec_ls)
    #
    # test_spec_rank = [[e, rank_dict[e+'-log']] for e in test_spec]
    # test_spec_rank.sort(key=lambda x:x[1])
    # for fn, num in test_spec_rank:
    #     print('%s\t%i' % (fn, num))
    #     # print('%s' % fn)

    for fn, num in solved_spec_ls + unsolved_spec_ls:
        print('%s\t%i' % (fn, num))


def baseline_time():
    def get_all_num(filepath):
        res = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()


                res.append([line.split(' ')[0], float(line.split(' ')[1])])
        return res

    cvc_num = get_all_num('../../benchmarks/cvc4_time.txt')
    baseline_num = get_all_num('../../benchmarks/baseline_time.txt')

    cvc_num.sort(key= lambda x:x[1])
    baseline_num.sort(key= lambda x:x[1])

    cvc_num = [e[1] for e in cvc_num]
    baseline_num = [e[1] for e in baseline_num]

    print(np.max(cvc_num), np.mean(cvc_num), np.median(cvc_num))
    print(np.max(baseline_num), np.mean(baseline_num), np.median(baseline_num))


def plot_curve():

    def get_all_num(filepath, sep='\t', trans=int):
        res = []
        with open(filepath, 'r') as f:
            for line in f:
                num = trans(line.split(sep)[1])
                if num == 0:
                    break
                res.append([line.split(sep)[0], trans(line.split(sep)[1])])
        return res

    num_ss = get_all_num('../../benchmarks/ss.txt')
    num_ss_g_embed = get_all_num('../../benchmarks/ss_g_embed_all.txt')
    num_ft_gs_embed = get_all_num('../../benchmarks/ft_gs_embed.txt')
    num_cvc = get_all_num('../../benchmarks/cvc4_time.txt', ' ', float)
    num_baseline = get_all_num('../../benchmarks/baseline_time.txt', ' ', float)
    num_eu = get_all_num('../../benchmarks/eusolver_time.txt', ' ', float)


    num_ss_g_embed.sort(key=lambda x:x[1])
    num_ft_gs_embed.sort(key=lambda x:x[1])
    num_cvc.sort(key=lambda x:x[1])
    num_baseline.sort(key=lambda x:x[1])
    num_eu.sort(key=lambda x: x[1])

    # num_ss_g_embed = np.array([e[1] for e in num_ss_g_embed])
    # num_ft_gs_embed = np.array([e[1] for e in num_ft_gs_embed])
    #
    # print(np.sum(num_ss_g_embed[:12]), np.sum(num_ft_gs_embed[:12]))
    # print(np.sum(num_ss_g_embed[:24]), np.sum(num_ft_gs_embed[:24]))
    # print(np.sum(num_ss_g_embed[:34]), np.sum(num_ft_gs_embed[:34]))

    # spdup_ls = [float(num_ss_g_embed[i][1]) / num_ft_gs_embed[i][1] for i in range(len(num_ss_g_embed))] + [100.0, 100.0]
    #
    # bin_ls = [0, 0, 0, 0, 0]
    # x_ls = [1, 2, 5, 10, 20]
    #
    # for e in spdup_ls:
    #     i = 0
    #     while i < len(x_ls) and e > x_ls[i]:
    #         i += 1
    #     bin_ls[i-1] += 1
    #
    # matplotlib.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    #
    # fig.set_size_inches(5, 3)
    #
    # rects = ax.bar([0, 0.5, 1, 1.5, 2], bin_ls, width=0.3)
    # ticks = ax.get_xticks()
    # ax.set_xticklabels([0, r'1~2$\times$', r'2~5$\times$', r'5~10$\times$', r'10~20$\times$', r'>20$\times$'])
    #
    # def autolabel(rects):
    #     """
    #     Attach a text label above each bar displaying its height
    #     """
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
    #                 '%d' % int(height),
    #                 ha='center', va='bottom')
    #
    # autolabel(rects)
    # ax.set_ylim([0, 12])
    # ax.set(xlabel='Speedup', ylabel='Numbers of instances')
    # # ax.grid()
    # plt.tight_layout()
    #
    # plt.show()
    #
    # for i in range(min(len(num_ss_g_embed), len(num_ft_gs_embed))):
    #     print('%s\t%s\t%i\t%i' % (num_ss_g_embed[i][0], num_ft_gs_embed[i][0],
    #                               num_ss_g_embed[i][1], num_ft_gs_embed[i][1]))

    # with open('../../benchmarks/test_nofalse', 'r') as ff:
    #     test_dict = set([line.strip() for line in ff])
    #
    # num_cvc = [e for e in num_cvc if e[0] in test_dict]
    # num_baseline =[e for e in num_baseline if e[0] in test_dict]

    num_ss_g_embed = np.array([e[1] / 3.5 for e in num_ss_g_embed])
    num_ft_gs_embed = np.array([e[1] / 4.0 for e in num_ft_gs_embed])
    num_cvc = np.array([e[1] for e in num_cvc])
    num_baseline = np.array([e[1] for e in num_baseline])
    num_eu = np.array([e[1] for e in num_eu])


    y_ss, y_ss_g_embed, y_ft_gs_embed, y_cvc, y_baseline, y_eu = [], [], [], [], [], []

    max_time = np.max([np.max(num_ss_g_embed), np.max(num_cvc), np.max(num_baseline), np.max(num_eu)])
    # max_time = np.max([np.max(num_ft_gs_embed), np.max(num_cvc), np.max(num_baseline)])

    x_ls = list(np.linspace(0, 1, 100))+list(range(1, int(max_time)+3000))

    for x in x_ls:
        y_ss_g_embed.append(np.sum(num_ss_g_embed <= x))
        # y_ft_gs_embed.append(np.sum(num_ft_gs_embed <= x))
        y_cvc.append(np.sum(num_cvc <= x))
        y_baseline.append(np.sum(num_baseline <= x))
        y_eu.append(np.sum(num_eu <= x))

    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()

    linewidth = 5
    ax.plot(y_ss_g_embed, x_ls, linewidth=linewidth, label='Out-of-Box solver')
    # ax.plot(y_ft_gs_embed, x_ls, label='Meta-Solver')
    ax.plot(y_cvc, x_ls, linewidth=linewidth, label='CVC4')
    ax.plot(y_baseline, x_ls, linewidth=linewidth, label='ESymbolic Solver')
    ax.plot(y_eu, x_ls, linewidth=linewidth, label='EUSolver')
    ax.legend()

    ax.set(xlabel='# instances solved', ylabel='Time (Seconds)')
    # ax.set_yscale('log')
    ax.grid()

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # parse_single_log('../../benchmarks/no_ft_gembed_single_log')
    # parse_single_log('../../a2c_multistep_single_log')

    # sort_spec('../../a2c_multistep_single_log')

    # sort_spec('../../benchmarks/ft_gembed_single_log')

    # sort_spec('../../benchmarks/ss_gs_embed859999_all_single_log')
    # parse_single_log('../../benchmarks/ss_gs_embed859999_all_single_log')

    # parse_single_log('../../benchmarks/no_ft_gembed_single_log')

    sort_spec('../../benchmarks/re_ss_g_embed_859999_all_single_log')
    # parse_single_log('../../benchmarks/re_ss_g_embed_859999_all_single_log')


    # sort_spec('../../a2c_onestep_single_log')

    # print('_______________')
    # sort_spec('../../benchmarks/ft_gembed8999_model150_single_log')

    # plot_curve()

    # baseline_time()

