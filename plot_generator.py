import matplotlib.pyplot as plt
import seaborn as sns
from Config.util import *


while True:
    path = input('file name: ')
    if path == 'end':
        break
    UB, UP, RB , line = load_f1_scores(F"Plots/{path}_f1_repeat.pkl")
    UB2, UP2, RB2 , line = load_f1_scores(F"Plots/{path}_f1_wide.pkl")
    average_f1_scores_UB = np.mean(UB, axis=0)
    average_f1_scores_UP = np.mean(UP, axis=0)
    average_f1_scores_RB = np.mean(RB, axis=0)
    average_f1_scores_UB2 = np.mean(UB2, axis=0)
    average_f1_scores_UP2 = np.mean(UP2, axis=0)
    average_f1_scores_RB2 = np.mean(RB2, axis=0)

    step, num_samples = 50, 800

    plt.figure(figsize=(12, 6))
    plt.plot(range(step, num_samples+1, step), average_f1_scores_UB, label='Warm-Start Policy', color='blue')
    plt.plot(range(step, num_samples+1, step), average_f1_scores_UP, label='Cold-Start Policy', color='orange')
    plt.plot(range(step, num_samples+1, step), average_f1_scores_RB, label='Random Selection Policy', color='green')
    plt.axhline(y = line, color = 'r', linestyle = 'dashed', label='High-Data Benchmark')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('F1 Score')
    plt.title('Comparison of F1 Scores with huge repetition for Credit dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(F'Images/{path}_repeat.png', bbox_inches='tight')
    # plt.show()

    num_samples = 10000

    plt.figure(figsize=(12, 6))
    plt.plot(range(step, num_samples+1, step), average_f1_scores_UB2, label='Warm-Start Policy', color='blue')
    plt.plot(range(step, num_samples+1, step), average_f1_scores_UP2, label='Cold-Start Policy', color='orange')
    plt.plot(range(step, num_samples+1, step), average_f1_scores_RB2, label='Random Selection Policy', color='green')
    plt.axhline(y = line, color = 'r', linestyle = 'dashed', label='High-Data Benchmark')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('F1 Score')
    plt.title('Comparison of F1 Scores with wide spectrum for Credit Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(F'Images/{path}_wide.png', bbox_inches='tight')
    # plt.show()