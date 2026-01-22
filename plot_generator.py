import matplotlib.pyplot as plt
from Config.util import *
import json

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
    
    # Load DopeWolfe results
    with open(f'Results/DopeWolf/{path}_dopewolfe_results.json', 'r') as f:
        dopewolfe_data = json.load(f)
    
    dopewolfe_mean = np.array(dopewolfe_data['DopeWolfe']['mean'])
    dopewolfe_std = np.array(dopewolfe_data['DopeWolfe']['std'])
    dopewolfe_all_runs = np.array(dopewolfe_data['DopeWolfe']['all_runs'])
    dopewolfe_ci = 1.96 * dopewolfe_std / np.sqrt(len(dopewolfe_all_runs))

    # Load GURO results
    with open(f'Results/GURO/{path}_guro_results.json', 'r') as f:
        guro_data = json.load(f)
    
    guro_mean = np.array(guro_data['GURO']['mean'])
    guro_std = np.array(guro_data['GURO']['std'])
    guro_all_runs = np.array(guro_data['GURO']['all_runs'])
    guro_ci = 1.96 * guro_std / np.sqrt(len(guro_all_runs))

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

    acc_max = acc_list.max()

    step, num_samples = 50, 800
    x = np.arange(step, num_samples + 1, step)

    # ---- Print first iteration accuracy for all methods ----
    print(f"\n{'='*50}")
    print(f"First Iteration Accuracy (at {step} training samples) - {path.capitalize()} Dataset")
    print(f"{'='*50}")
    print(f"  Warm-Start Policy:       {mean_UB[0]:.4f} ± {ci_UB[0]:.4f}")
    print(f"  Cold-Start Policy:      {mean_UP[0]:.4f} ± {ci_UP[0]:.4f}")
    print(f"  Random Selection:        {mean_RB[0]:.4f} ± {ci_RB[0]:.4f}")
    print(f"  DopeWolfe:               {dopewolfe_mean[0]:.4f} ± {dopewolfe_ci[0]:.4f}")
    print(f"  GURO:                   {guro_mean[0]:.4f} ± {guro_ci[0]:.4f}")
    print(f"{'='*50}\n")

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
    
    # ---- DopeWolfe ----
    plt.plot(x, dopewolfe_mean, label='DopeWolfe', color='purple')
    plt.fill_between(x, dopewolfe_mean - dopewolfe_ci, dopewolfe_mean + dopewolfe_ci,
                     color='purple', alpha=0.2)

    # ---- GURO ----
    plt.plot(x, guro_mean, label='GURO', color='brown')
    plt.fill_between(x, guro_mean - guro_ci, guro_mean + guro_ci,
                     color='brown', alpha=0.2)

    # ---- Practical performance limit (acc_max) ----
    plt.axhline(y=acc_max, color='red', linestyle='dashed',
                label='Practical Performance Limit')

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Data Performance')
    plt.title(f'{path.capitalize()} Dataset')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'Images/{path}.png', bbox_inches='tight')
    # plt.show()