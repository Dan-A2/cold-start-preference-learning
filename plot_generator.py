import matplotlib.pyplot as plt
from Config.util import *

while True:
    path = input('file name: ')
    if path == 'end':
        break

    # Now acc_list instead of acc
    UB, UP, RB, acc_list, _ = load_accs(f"Plots/{path}_acc.pkl")

    UB = np.array(UB)
    UP = np.array(UP)
    RB = np.array(RB)
    acc_list = np.array(acc_list)

    # ---- statistics helper ----
    def mean_ci(x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        ci = 1.96 * std / np.sqrt(x.shape[0])
        return mean, ci

    # ---- compute means and CIs ----
    mean_UB, ci_UB = mean_ci(UB)
    mean_UP, ci_UP = mean_ci(UP)
    mean_RB, ci_RB = mean_ci(RB)

    acc_mean = acc_list.mean()
    acc_ci = 1.96 * acc_list.std() / np.sqrt(len(acc_list))

    step, num_samples = 50, 800
    x = np.arange(step, num_samples + 1, step)

    plt.figure(figsize=(12, 6))

    # ---- UB ----
    plt.plot(x, mean_UB, label='Warm-Start Policy', color='blue')
    plt.fill_between(x, mean_UB - ci_UB, mean_UB + ci_UB,
                     color='blue', alpha=0.2)

    # ---- UP ----
    plt.plot(x, mean_UP, label='Cold-Start Policy', color='orange')
    plt.fill_between(x, mean_UP - ci_UP, mean_UP + ci_UP,
                     color='orange', alpha=0.2)

    # ---- RB ----
    plt.plot(x, mean_RB, label='Random Selection Policy', color='green')
    plt.fill_between(x, mean_RB - ci_RB, mean_RB + ci_RB,
                     color='green', alpha=0.2)

    # ---- Practical performance limit (acc_list) ----
    plt.axhline(y=acc_mean, color='red', linestyle='dashed',
                label='Practical Performance Limit')
    plt.fill_between(x,
                     acc_mean - acc_ci,
                     acc_mean + acc_ci,
                     color='red', alpha=0.15)

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Data Performance')
    plt.title(f'{path.capitalize()} Dataset')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'Images/{path}.png', bbox_inches='tight')
    # plt.show()