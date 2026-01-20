"""
Pairwise Preference Learning with DopeWolfe Active Selection and XGBoost

This script implements: 
- DopeWolfe (Frank-Wolfe) algorithm for selecting informative pairwise queries
- XGBoost for learning pairwise preferences
- Step-by-step evaluation on a fixed test set
"""

from Config.util import *
import json
import time
from sklearn.linear_model import LogisticRegression


def train_linear_pairwise(X_train:  np.ndarray, y_train: np. ndarray) -> LogisticRegression: 
    """
    Train a logistic regression classifier for pairwise preference prediction.
    """
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate model accuracy on test set."""
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy


def run_experiment(df: pd.DataFrame, target_col: str, test_df: pd.DataFrame,
                   total_pairs: int = 800, step:  int = 50, 
                   repeats: int = 40) -> dict:
    """
    Run the full experiment on DopeWolfe
    
    Parameters:
        df: dataframe with features and target
        target_col:  name of target column
        test_df: test dataframe with paired features and labels
        total_pairs: total number of pairs to sample
        step: number of pairs to add at each step
        repeats:  number of experiment repetitions
    
    Returns:
        Dictionary with accuracy results for each method
    """
    # Prepare test data
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    
    # Number of evaluation steps
    num_steps = total_pairs // step
    eval_points = np.arange(step, total_pairs + 1, step)
    
    # Store results for each method
    results = {
        'DopeWolfe': np.zeros((repeats, num_steps)),
    }
    
    # Store timing for each method (total time per repeat)
    timing = {
        'DopeWolfe': np.zeros(repeats),
    }
    
    for rep in range(repeats):
        print(f"Repeat {rep + 1}/{repeats}", end=" ")
        
        # Generate a pool of candidate pairs (larger than total_pairs needed)
        candidate_pairs = generate_random_pairs(df, total_pairs * 3)
        
        for method in ['DopeWolfe']:
            print(f"[{method}]", end=" ")
            
            # Track selected pairs
            selected_pairs = []
            remaining_candidates = candidate_pairs.copy()
            
            # Track total algorithm time for this repeat
            total_algo_time = 0.0
            
            for step_idx, n_pairs in enumerate(eval_points):
                # Number of new pairs to select in this step
                n_new = step if step_idx == 0 else step
                
                if len(remaining_candidates) == 0:
                    break
                
                # Select new pairs based on method
                if method == 'DopeWolfe':
                    start_time = time.time()
                    new_pairs = select_pairs_dopewolfe(
                        df, remaining_candidates, target_col, n_new, use_randomized=True
                    )
                    total_algo_time += time.time() - start_time
                elif method == 'Dope':
                    start_time = time.time()
                    new_pairs = select_pairs_dopewolfe(
                        df, remaining_candidates, target_col, n_new, use_randomized=False
                    )
                    total_algo_time += time.time() - start_time
                
                # Add new pairs to selected set
                selected_pairs.extend(new_pairs)
                
                # Remove selected pairs from candidates
                remaining_candidates = [p for p in remaining_candidates if p not in new_pairs]
                
                # Create training data from selected pairs
                df_paired = create_pair_df(df, selected_pairs, target_col)
                X_train = df_paired.drop(columns=['label']).values
                y_train = df_paired['label'].values
                
                # Train Linear model
                model = train_linear_pairwise(X_train, y_train)
                
                # Evaluate on test set
                accuracy = evaluate_model(model, X_test, y_test)
                results[method][rep, step_idx] = accuracy
                
            timing[method][rep] = total_algo_time
            
        print()
    
    return results, eval_points, timing


if __name__ == "__main__":
    # Configuration
    NAME = "fifa"
    DATASET_PATH = 'Datasets/final_data.csv'
    TARGET_COL = 'value_eur'
    TOTAL_PAIRS = 800
    STEP = 50
    REPEATS = 40
    TEST_SIZE = 20000
    
    print("=" * 60)
    print("Pairwise Preference Learning with DopeWolfe + Linear Model")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    data = pd.read_csv(DATASET_PATH)
    
    
    
    data = standardize_features(data, TARGET_COL)
    print(f"    Data shape: {data.shape}")
    print(f"    Target column: {TARGET_COL}")
    
    # Generate test pairs and create test dataframe
    print(f"\n[2] Generating {TEST_SIZE} test pairs...")
    test_pairs = generate_random_pairs(data, TEST_SIZE)
    test_df = create_pair_df(data, test_pairs, TARGET_COL)
    print(f"    Test dataframe shape: {test_df.shape}")
    
    # Run experiment
    print(f"\n[3] Running experiment...")
    print(f"    Total pairs: {TOTAL_PAIRS}")
    print(f"    Step size: {STEP}")
    print(f"    Repeats:  {REPEATS}")
    print("-" * 60)
    
    results, eval_points, timing = run_experiment(
        df=data,
        target_col=TARGET_COL,
        test_df=test_df,
        total_pairs=TOTAL_PAIRS,
        step=STEP,
        repeats=REPEATS
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Print timing results
    print("\nAlgorithm Execution Time (seconds):")
    for method in ['DopeWolfe']:
        mean_time = timing[method].mean()
        std_time = timing[method].std() / np.sqrt(len(timing[method]))
        print(f"    {method}: {mean_time:.3f} ± {std_time:.3f} seconds per run")

    
    for method in ['DopeWolfe']:
        final_acc = results[method][: , -1]
        print(f"\n{method}:")
        print(f"    Final accuracy: {final_acc.mean():.4f} ± {final_acc.std() / np.sqrt(len(final_acc)):.4f}")
        
        # Print accuracy at each evaluation point
        print(f"    Step-by-step accuracy:")
        for i, n_pairs in enumerate(eval_points):
            acc = results[method][:, i]
            print(f"        {n_pairs: 4d} pairs: {acc.mean():.4f} ± {acc.std() / np.sqrt(len(acc)):.4f}")
    
    # Save results to JSON
    print("\n[5] Saving results...")

    results_json = {
        'n_pairs': eval_points.tolist(),
        'DopeWolfe':  {
            'mean':  results['DopeWolfe'].mean(axis=0).tolist(),
            'std': results['DopeWolfe'].std(axis=0).tolist(),
            'all_runs': results['DopeWolfe'].tolist(),
            'timing_seconds': {
                'mean': float(timing['DopeWolfe'].mean()),
                'std': float(timing['DopeWolfe'].std()),
                'all_runs':  timing['DopeWolfe'].tolist()
            }
        }
    }

    with open(f'Results/{NAME}_dopewolfe_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"    Results saved to 'Results/{NAME}_dopewolfe_results.json'")
    
    print("\nDone!")