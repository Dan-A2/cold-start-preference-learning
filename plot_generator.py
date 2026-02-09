import matplotlib.pyplot as plt
from Config.util import *
import json

while True:
    path = input('file name: ')
    if path == 'end':
        break

    # Load XGBoost results
    UB, UP, RB, acc_list, _ = load_accs(f"Plots/{path}_acc.pkl")
    UB = np.array(UB)
    UP = np.array(UP)
    RB = np.array(RB)
    acc_list = np.array(acc_list)
    
    # Load DopeWolfe results (non-pretrained)
    with open(f'Results/DopeWolf/{path}_dopewolfe_results.json', 'r') as f:
        dopewolfe_data = json.load(f)

    dopewolfe_mean = np.array(dopewolfe_data['DopeWolfe']['mean'])
    dopewolfe_std = np.array(dopewolfe_data['DopeWolfe']['std'])
    dopewolfe_all_runs = np.array(dopewolfe_data['DopeWolfe']['all_runs'])
    dopewolfe_ci = 1.96 * dopewolfe_std / np.sqrt(len(dopewolfe_all_runs))

    # Load DopeWolfe results (pretrained)
    with open(f'Results/DopeWolf/{path}_dopewolfe_results_pretrained.json', 'r') as f:
        dopewolfe_pretrain_data = json.load(f)

    dopewolfe_pretrain_mean = np.array(dopewolfe_pretrain_data['DopeWolfe']['mean'])
    dopewolfe_pretrain_std = np.array(dopewolfe_pretrain_data['DopeWolfe']['std'])
    dopewolfe_pretrain_all_runs = np.array(dopewolfe_pretrain_data['DopeWolfe']['all_runs'])
    dopewolfe_pretrain_ci = 1.96 * dopewolfe_pretrain_std / np.sqrt(len(dopewolfe_pretrain_all_runs))

    # Load GURO results (non-pretrained)
    with open(f'Results/GURO/{path}_guro_results.json', 'r') as f:
        guro_data = json.load(f)

    guro_mean = np.array(guro_data['GURO']['mean'])
    guro_std = np.array(guro_data['GURO']['std'])
    guro_all_runs = np.array(guro_data['GURO']['all_runs'])
    guro_ci = 1.96 * guro_std / np.sqrt(len(guro_all_runs))

    # Load GURO results (pretrained)
    with open(f'Results/GURO/{path}_guro_results_pretrained.json', 'r') as f:
        guro_pretrain_data = json.load(f)
    
    guro_pretrain_mean = np.array(guro_pretrain_data['GURO']['mean'])
    guro_pretrain_std = np.array(guro_pretrain_data['GURO']['std'])
    guro_pretrain_all_runs = np.array(guro_pretrain_data['GURO']['all_runs'])
    guro_pretrain_ci = 1.96 * guro_pretrain_std / np.sqrt(len(guro_pretrain_all_runs))

    # Load regression max accuracy
    with open(f'Results/Regression/{path}.pkl', 'rb') as f:
        acc_list_regression = pickle.load(f)
    acc_list_regression = np.array(acc_list_regression)
    acc_max_regression = acc_list_regression.max()

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

    acc_max_xgb = acc_list.max()

    step, num_samples = 50, 800
    x = np.arange(step, num_samples + 1, step)

    # ---- Print first iteration accuracy for all methods ----
    print(f"\n{'='*60}")
    print(f"First Iteration Accuracy (at {step} training samples)")
    print(f"Dataset: {path.capitalize()}")
    print(f"{'='*60}")
    print(f"\nXGBoost Models:")
    print(f"  Warm-Start Policy:       {mean_UB[0]:.4f} ± {ci_UB[0]:.4f}")
    print(f"  Cold-Start Policy:       {mean_UP[0]:.4f} ± {ci_UP[0]:.4f}")
    print(f"  Random Selection:        {mean_RB[0]:.4f} ± {ci_RB[0]:.4f}")
    print(f"\nRegression Models:")
    print(f"  DopeWolfe:               {dopewolfe_mean[0]:.4f} ± {dopewolfe_ci[0]:.4f}")
    print(f"  DopeWolfe (Pretrained):  {dopewolfe_pretrain_mean[0]:.4f} ± {dopewolfe_pretrain_ci[0]:.4f}")
    print(f"  GURO:                    {guro_mean[0]:.4f} ± {guro_ci[0]:.4f}")
    print(f"  GURO (Pretrained):       {guro_pretrain_mean[0]:.4f} ± {guro_pretrain_ci[0]:.4f}")
    print(f"\nPerformance Limits:")
    print(f"  XGBoost Max:             {acc_max_xgb:.4f}")
    print(f"  Regression Max:          {acc_max_regression:.4f}")
    print(f"{'='*60}\n")

    # ============================================================================
    # PLOT 1: XGBoost Models
    # ============================================================================
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # ---- UB ----
    ax1.plot(x, mean_UB, label='Warm-Start Policy', color='blue', linewidth=2, linestyle='dashed')
    ax1.fill_between(x, mean_UB - ci_UB, mean_UB + ci_UB,
                     color='blue', alpha=0.2)

    # ---- UP ----
    ax1.plot(x, mean_UP, label='Cold-Start Policy', color='orange', linewidth=2)
    ax1.fill_between(x, mean_UP - ci_UP, mean_UP + ci_UP,
                     color='orange', alpha=0.2)

    # ---- RB ----
    ax1.plot(x, mean_RB, label='Random Selection Policy', color='green', linewidth=2, linestyle='dashed')
    ax1.fill_between(x, mean_RB - ci_RB, mean_RB + ci_RB,
                     color='green', alpha=0.2)

    # ---- Practical performance limit (XGBoost) ----
    ax1.axhline(y=acc_max_xgb, color='red', linestyle='dashdot',
                label='Practical Performance Limit', linewidth=2)

    ax1.set_xlabel('Number of Training Samples', fontsize=24)
    ax1.set_ylabel('Test Data Performance', fontsize=24)
    ax1.set_title(f'{"FIFA" if path == 'fifa' else path.capitalize()} Dataset - XGBoost Models', fontsize=28, fontweight='bold')
    ax1.legend(fontsize=20, loc='best')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'Images/{path}_xgboost.png', bbox_inches='tight', dpi=300)
    print(f"Saved: Images/{path}_xgboost.png")
    # plt.show()
    plt.close()

    # ============================================================================
    # PLOT 2: Regression Models
    # ============================================================================
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # ---- DopeWolfe ----
    ax2.plot(x, dopewolfe_mean, label='DopeWolfe', color='#8731CE', linewidth=2, linestyle='dashed')
    ax2.fill_between(x, dopewolfe_mean - dopewolfe_ci, dopewolfe_mean + dopewolfe_ci,
                     color='#8731CE', alpha=0.2)

    # ---- DopeWolfe Pretrained ----
    ax2.plot(x, dopewolfe_pretrain_mean, label='DopeWolfe (cold-start)', 
             color='#8731CE', linewidth=2)
    ax2.fill_between(x, dopewolfe_pretrain_mean - dopewolfe_pretrain_ci, 
                     dopewolfe_pretrain_mean + dopewolfe_pretrain_ci,
                     color='#8731CE', alpha=0.15)

    # ---- GURO ----
    ax2.plot(x, guro_mean, label='GURO', color='#78CE31', linewidth=2, linestyle='dashed')
    ax2.fill_between(x, guro_mean - guro_ci, guro_mean + guro_ci,
                     color='#78CE31', alpha=0.2)

    # ---- GURO Pretrained ----
    ax2.plot(x, guro_pretrain_mean, label='GURO (cold-start)', 
             color='#78CE31', linewidth=2)
    ax2.fill_between(x, guro_pretrain_mean - guro_pretrain_ci, 
                     guro_pretrain_mean + guro_pretrain_ci,
                     color='#78CE31', alpha=0.15)

    # ---- Practical performance limit (Regression) ----
    ax2.axhline(y=acc_max_regression, color='red', linestyle='dashdot',
                label='Practical Performance Limit', linewidth=2)

    ax2.set_xlabel('Number of Training Samples', fontsize=24)
    ax2.set_ylabel('Test Data Performance', fontsize=24)
    ax2.set_title(f'{"FIFA" if path == 'fifa' else path.capitalize()} Dataset - Logistic Regression Models', fontsize=28, fontweight='bold')
    ax2.legend(fontsize=20, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'Images/{path}_regression.png', bbox_inches='tight', dpi=300)
    print(f"Saved: Images/{path}_regression.png")
    # plt.show()
    plt.close()

print("\nAll plots generated successfully!")