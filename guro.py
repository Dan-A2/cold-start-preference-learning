"""
Pairwise Preference Learning with GURO Algorithm
- GURO (Greedy Uncertainty Reduction for Ordering) for selecting informative pairwise queries
- Logistic Regression for learning pairwise preferences
- Step-by-step evaluation on a fixed test set
"""

from Config.util import *
import json
import time


def evaluate_test_accuracy(algo, X, test_pairs, true_order):
    """
    Evaluate pairwise prediction accuracy on test set. 
    
    Parameters:
        algo: trained algorithm with theta_hat
        X: feature matrix
        test_pairs: list of (i, j) pairs to evaluate
        true_order: ground truth ordering based on target values
    
    Returns:
        accuracy: proportion of correctly predicted pairwise comparisons
    """
    correct = 0
    total = len(test_pairs)
    
    for i, j in test_pairs: 
        # Get predicted comparison
        diff = X[i] - X[j]
        pred_prob = expit(diff.dot(algo.theta_hat))
        pred_label = 1 if pred_prob > 0.5 else 0
        
        # True label based on ordering
        true_label = 1 if true_order[i] > true_order[j] else 0
        
        if pred_label == true_label:
            correct += 1
    
    return correct / total


def run_experiment(df: pd.DataFrame, target_col: str, test_pairs: list,
                   total_pairs: int = 800, step: int = 50,
                   repeats: int = 40, pretrained_model=None,
                   pretrained_data=None) -> dict:
    """
    Run the full experiment comparing GURO
    """
    # Prepare feature matrix and true ordering
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values
    true_order = rankdata(y)
    
    n = X.shape[0]
    
    # Number of evaluation steps
    num_steps = total_pairs // step
    eval_points = np.arange(step, total_pairs + 1, step)
    
    # Store results for each method
    results = {
        'GURO': np.zeros((repeats, num_steps))
    }
    
    # Store timing for each method
    timing = {
        'GURO': np.zeros(repeats)
    }
    
    for rep in tqdm(range(repeats), desc="Running experiments"):
        seed = rep + 100
        
        # Generate pool of candidate pairs for this repeat
        rng = np.random.RandomState(seed)
        all_combinations = list(itertools.combinations(range(n), 2))
        rng.shuffle(all_combinations)
        candidate_pairs = all_combinations[:total_pairs * 3]
        
        for method in ['GURO']:
            # Initialize algorithm
            algo = GURORealData(
                X,
                available_combinations=candidate_pairs.copy(),
                seed=seed,
                sample_combinations=True,
                update_every=10
            )
            
            # Initialize with pretrained model if provided
            if pretrained_model is not None:
                algo.theta_hat = pretrained_model.coef_[0].copy()
                algo.model = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER, warm_start=True)
                algo.model.coef_ = pretrained_model.coef_.copy()
                algo.model.intercept_ = pretrained_model.intercept_.copy()
                algo.model.classes_ = pretrained_model.classes_.copy()
                
                # Initialize obs_data and obs_labels with pretrained data
                if pretrained_data is not None:
                    X_pretrain, y_pretrain = pretrained_data
                    # Convert to list of differences for GURO's internal storage
                    algo.obs_data = X_pretrain.tolist()
                    algo.obs_labels = y_pretrain.tolist()
            
            # Track algorithm time
            total_algo_time = 0.0
            
            current_step = 0
            step_idx = 0
            
            for t in range(total_pairs):
                # Time the action selection (this is the algorithm part)
                start_time = time.time()
                action, _ = algo.act()
                total_algo_time += time.time() - start_time
                
                i, j = action
                
                # Get true label based on ordering
                label = 1 if true_order[i] > true_order[j] else 0
                
                # Update algorithm (not timed - this is just bookkeeping + model update)
                algo.update(i, j, label)
                
                current_step += 1
                
                # Evaluate at each step point
                if current_step in eval_points: 
                    accuracy = evaluate_test_accuracy(algo, X, test_pairs, true_order)
                    results[method][rep, step_idx] = accuracy
                    step_idx += 1
            
            timing[method][rep] = total_algo_time
    
    return results, eval_points, timing


if __name__ == "__main__": 
    # Configuration
    NAME = "student"
    DATASET_PATH = 'Datasets/Student_performance_data _.csv'
    TARGET_COL = 'GPA'
    USE_PRETRAINED = True
    TOTAL_PAIRS = 800
    STEP = 50
    REPEATS = 40
    TEST_SIZE = 20000
    
    print("=" * 60)
    print("Pairwise Preference Learning with GURO + Linear Model")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    data = pd.read_csv(DATASET_PATH)
    
    # Cleaning code (if needed)
    data.drop(columns=['GradeClass', 'StudentID'], inplace=True)
    object_columns = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring',
                'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
    numeric_columns = ['Age', 'StudyTimeWeekly', 'Absences']
    object_columns.remove('Tutoring')
    numeric_columns.append('Tutoring')
    data[numeric_columns] = data[numeric_columns].astype(int)
    data[object_columns] = data[object_columns].astype(str)
    data = pd.get_dummies(data, columns=object_columns)
    data = standardize_features(data, TARGET_COL)
    
    # PCA
    X = data.drop(columns=[TARGET_COL])
    pca = PCA(n_components=2)
    pca.fit(X)
    data['PCA'] = pca.transform(X)[:, 0]
    
    print(f"    Data shape: {data.shape}")
    print(f"    Target column: {TARGET_COL}")
    
    # Pretrain model if needed
    pretrained_model_pca = None
    pretrained_data = None
    
    if USE_PRETRAINED:
        
        # Variance calculation to determine number of pairs
        var, residuals = calculate_pca_var(data, TARGET_COL)
        max_pairs = len(data) // 10
        alpha = 1e-6
        num_pairs = int(max_pairs / (1 + alpha * var))
        
        # Generate pretrained pairs with differences (not concatenated)
        pretrain_pairs_pca = generate_weighted_pairs(data, n=num_pairs, residuals=residuals)
        
        # Create difference-based features for PCA
        feature_cols = [col for col in data.columns if col != TARGET_COL]
        X_full = data[feature_cols].values
        
        X_diff_list = []
        y_diff_list = []
        
        for idx1, idx2 in pretrain_pairs_pca:
            # Use PCA-based label
            pca_val1 = data.iloc[idx1]['PCA']  # Assuming you have PCA column
            pca_val2 = data.iloc[idx2]['PCA']
            label = 1 if pca_val1 > pca_val2 else 0
            
            # Create difference
            diff = X_full[idx1] - X_full[idx2]
            X_diff_list.append(diff)
            y_diff_list.append(label)
        
        X_diff = np.array(X_diff_list)
        y_diff = np.array(y_diff_list)
        
        # Train PCA-compatible model
        pretrained_model_pca = LogisticRegression(
            max_iter=MAX_ITER,
            solver=SOLVER,
            warm_start=True
        )
        pretrained_model_pca.fit(X_diff, y_diff)
    
    # Generate test pairs
    print(f"\n[2] Generating {TEST_SIZE} test pairs...")
    n = len(data)
    np.random.seed(42)
    all_test_combinations = list(itertools.combinations(range(n), 2))
    np.random.shuffle(all_test_combinations)
    test_pairs = all_test_combinations[:TEST_SIZE]
    print(f"    Number of test pairs: {len(test_pairs)}")
    
    # Run experiment
    print(f"\n[3] Running experiment...")
    print(f"    Total pairs: {TOTAL_PAIRS}")
    print(f"    Step size: {STEP}")
    print(f"    Repeats: {REPEATS}")
    print("-" * 60)
    
    results, eval_points, timing = run_experiment(
        df=data,
        target_col=TARGET_COL,
        test_pairs=test_pairs,
        total_pairs=TOTAL_PAIRS,
        step=STEP,
        repeats=REPEATS,
        pretrained_model=pretrained_model_pca,
        pretrained_data=pretrained_data
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Print timing results
    print("\nAlgorithm Execution Time (seconds):")
    for method in ['GURO']:
        mean_time = timing[method].mean()
        std_time = timing[method].std() / np.sqrt(len(timing[method]))
        print(f"    {method}: {mean_time:.3f} ± {std_time:.3f} seconds per run")
    
    for method in ['GURO']:
        final_acc = results[method][:, -1]
        print(f"\n{method}:")
        print(f"    Final accuracy: {final_acc.mean():.4f} ± {final_acc.std() / np.sqrt(len(final_acc)):.4f}")
        
        # Print accuracy at each evaluation point
        print(f"    Step-by-step accuracy:")
        for i, n_pairs in enumerate(eval_points):
            acc = results[method][:, i]
            print(f"        {n_pairs:4d} pairs: {acc.mean():.4f} ± {acc.std() / np.sqrt(len(acc)):.4f}")
    
    # Save results to JSON
    print("\n[4] Saving results...")
    
    results_json = {
        'n_pairs': eval_points.tolist(),
        'GURO': {
            'mean': results['GURO'].mean(axis=0).tolist(),
            'std': results['GURO'].std(axis=0).tolist(),
            'all_runs': results['GURO'].tolist(),
            'timing_seconds': {
                'mean': float(timing['GURO'].mean()),
                'std': float(timing['GURO'].std()),
                'all_runs': timing['GURO'].tolist()
            }
        }
    }
    
    path = f'Results/GURO/{NAME}_guro_results.json'
    if USE_PRETRAINED:
        path = f'Results/GURO/{NAME}_guro_results_pretrained.json'
    
    with open(path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"    Results saved to '{path}'")
    
    print("\nDone!")