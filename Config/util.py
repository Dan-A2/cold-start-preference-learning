import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tqdm import tqdm
import pickle
import itertools


XGB_ESTIMATORS = 500
MAX_ITER = 1000
SOLVER = 'lbfgs'


def encode_object_columns(df, columns):
    '''
    This function converts given columns of a dataframe to integers
    using pytorch label encoder
    '''
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


def is_consumption(df, target_column, label_column, threshold=0.5):
    '''
    This function determines if a variable is innately categorical or numeric
    '''
    # Ensure input column exists in the dataframe
    if target_column not in df.columns or label_column not in df.columns:
        raise ValueError("Target or label column not found in the dataframe.")

    # Extract numerical columns excluding the label column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(label_column)

    # Extract unique values from the target column
    unique_values = df[target_column].unique()

    # Initialize StandardScaler for normalization
    scaler = StandardScaler()

    for value in unique_values:
        # Create two subsets: one without the current value, and one only with the current value
        subset_without_value = df[df[target_column] != value][numeric_cols]
        subset_with_value = df[df[target_column] == value][numeric_cols]

        # Standardize both subsets
        subset_without_value_scaled = scaler.fit_transform(subset_without_value)
        subset_with_value_scaled = scaler.fit_transform(subset_with_value)

        # Fit a 1-dimensional PCA to each subset
        pca_without_value = PCA(n_components=1)
        pca_with_value = PCA(n_components=1)

        pca_without_value.fit(subset_without_value_scaled)
        pca_with_value.fit(subset_with_value_scaled)

        # Get the principal components (eigenvectors)
        eigenvector_without_value = pca_without_value.components_[0]
        eigenvector_with_value = pca_with_value.components_[0]

        # Compute the cosine similarity between the eigenvectors
        cosine_similarity = np.dot(eigenvector_without_value, eigenvector_with_value) / (
            np.linalg.norm(eigenvector_without_value) * np.linalg.norm(eigenvector_with_value)
        )

        # Convert cosine similarity to cosine distance
        cosine_distance = 1 - cosine_similarity

        # Check if the distance exceeds the threshold
        if cosine_distance > threshold:
            return False  # The column is non-consumption (categorical)

    # If no high distances found, consider it a consumption variable
    return True


def standardize_features(df, target_col_name):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if target_col_name in numeric_cols:
        numeric_cols.remove(target_col_name)
    if 'GPT' in numeric_cols:
        numeric_cols.remove('GPT')
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def drop_redundant_cols(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)


def add_gpt_rank(df, ranking_csv_path):
    ranking = pd.read_csv(ranking_csv_path)
    ranking = ranking.sort_values("row_id").reset_index(drop=True)
    new_df = df.copy().reset_index(drop=True)
    new_df["GPT"] = ranking["rank"].values
    return new_df


def create_pair_pca(df, pairs, target_col):
    '''
    This function creates the dual dataframe used for our model according to the PCA
    '''
    paired_data = []
    df_tmp = df.drop(columns=[target_col])
    for idx1, idx2 in pairs:
        player1 = df_tmp.iloc[idx1].add_suffix('_1')
        player2 = df_tmp.iloc[idx2].add_suffix('_2')
        label = int(player1['PCA_1'] > player2['PCA_2'])
        pair = pd.concat([player1, player2])
        pair['label'] = label
        paired_data.append(pair)
    paired_df = pd.DataFrame(paired_data)
    return paired_df


def create_pair_gpt(df, pairs, target_col):
    '''
    This function creates the dual dataframe used for our model according to the GPT
    '''
    paired_data = []
    df_tmp = df.drop(columns=[target_col])
    for idx1, idx2 in pairs:
        player1 = df_tmp.iloc[idx1].add_suffix('_1')
        player2 = df_tmp.iloc[idx2].add_suffix('_2')
        label = int(player1['GPT_1'] > player2['GPT_2'])
        pair = pd.concat([player1, player2])
        pair['label'] = label
        paired_data.append(pair)
    paired_df = pd.DataFrame(paired_data)
    paired_df.drop(columns=['GPT_1', 'GPT_2'], inplace=True)
    return paired_df


def create_pair_df(df, pairs, target_col):
    '''
    This function creates the dual dataframe used for our model according to the actual value
    '''
    paired_data = []
    df_tmp = df.drop(columns=[target_col])
    for idx1, idx2 in pairs:
        player1 = df_tmp.iloc[idx1].add_suffix('_1')
        player2 = df_tmp.iloc[idx2].add_suffix('_2')
        label = int(df.iloc[idx1][target_col] > df.iloc[idx2][target_col])
        pair = pd.concat([player1, player2])
        pair['label'] = label
        paired_data.append(pair)
    paired_df = pd.DataFrame(paired_data)
    return paired_df


def noisy_price(p, relative_variance=0.05):
    noise = np.random.rand() * 2 - 1
    noise = noise * relative_variance
    return p * (1 + noise)


def create_pair_noisy(df, pairs, target_col, variance=0.05):
    '''
    This function creates the dual dataframe used for our model according to the actual label with ±5% noise
    '''
    paired_data = []
    df_tmp = df.drop(columns=[target_col])
    for idx1, idx2 in pairs:
        player1 = df_tmp.iloc[idx1].add_suffix('_1')
        player2 = df_tmp.iloc[idx2].add_suffix('_2')
        p1 = df.iloc[idx1][target_col]
        p2 = df.iloc[idx2][target_col]
        p1 = noisy_price(p1, relative_variance=variance)
        p2 = noisy_price(p2, relative_variance=variance)
        label = 1 if p1 > p2 else 0
        pair = pd.concat([player1, player2])
        pair['label'] = label
        paired_data.append(pair)
    paired_df = pd.DataFrame(paired_data)
    return paired_df


def create_pair_bradley(df, pairs, target_col):
    '''
    This function creates the dual dataframe used for our model according to the actual value with bradley-terry model
    '''
    paired_data = []
    df_tmp = df.drop(columns=[target_col])
    for idx1, idx2 in pairs:
        player1 = df_tmp.iloc[idx1].add_suffix('_1')
        player2 = df_tmp.iloc[idx2].add_suffix('_2')
        p1 = df.iloc[idx1][target_col]
        p2 = df.iloc[idx2][target_col]
        if p1 > p2:
            prob = p1 / (p1 + p2)
        else:
            prob = p2 / (p1 + p2)
        label = np.random.choice([1, 0], p=[prob, 1 - prob]) if p1 > p2 else np.random.choice([0, 1], p=[prob, 1 - prob])
        pair = pd.concat([player1, player2])
        pair['label'] = label
        paired_data.append(pair)
    paired_df = pd.DataFrame(paired_data)
    return paired_df


def create_pair_bradley_exp(df, pairs, target_col):
    '''
    This function creates the dual dataframe used for our model according to the actual value with exponential bradley-terry model
    '''
    paired_data = []
    df_tmp = df.drop(columns=[target_col])
    for idx1, idx2 in pairs:
        player1 = df_tmp.iloc[idx1].add_suffix('_1')
        player2 = df_tmp.iloc[idx2].add_suffix('_2')
        p1 = df.iloc[idx1][target_col]
        p2 = df.iloc[idx2][target_col]
        if p1 > p2:
            prob = np.exp(p1) / (np.exp(p1) + np.exp(p2))
        else:
            prob = np.exp(p2) / (np.exp(p1) + np.exp(p2))
        label = np.random.choice([1, 0], p=[prob, 1 - prob]) if p1 > p2 else np.random.choice([0, 1], p=[prob, 1 - prob])
        pair = pd.concat([player1, player2])
        pair['label'] = label
        paired_data.append(pair)
    paired_df = pd.DataFrame(paired_data)
    return paired_df


def generate_random_pairs(df, n):
    '''
    This function randomly selects 2 rows of the given dataframe
    '''
    pairs = []
    for _ in range(n):
        idx1, idx2 = np.random.choice(df.index, 2, replace=False)
        pairs.append((idx1, idx2))
    return pairs


def generate_weighted_pairs(df, n, residuals):
    '''
    This function selects 2 rows from the given dataframe with probabilities proportional to the residuals.
    Parameters:
        df (pd.DataFrame): The dataframe to select from.
        n (int): Number of pairs to generate.
        residuals (np.array): Residuals from the PCA-target line.
    Returns:
        list: A list of pairs of indexes.
    '''
    # Normalize residuals to create probabilities
    probabilities = np.abs(residuals) / np.sum(np.abs(residuals))
    
    pairs = []
    for _ in range(n):
        # Choose two indices based on the calculated probabilities
        idx1, idx2 = np.random.choice(df.index, 2, replace=False, p=probabilities)
        pairs.append((idx1, idx2))
    
    return pairs


def generate_dmatrix(df):
    X_pretrain = df.drop(columns=['label'])
    y_pretrain = df['label']
    return xgb.DMatrix(X_pretrain, label=y_pretrain, enable_categorical=True)


def pretrain_model(df, n_samples, train_params, target_col):
    '''
    This function pretrains the model on n samples using the PCA score
    '''
    pretrain_pairs = generate_random_pairs(df, n=n_samples)
    pretrain_df = create_pair_pca(df, pretrain_pairs, target_col)
    dtrain_pretrain = generate_dmatrix(pretrain_df)
    pretrained_model = xgb.train(train_params, dtrain_pretrain, num_boost_round=XGB_ESTIMATORS)
    return pretrained_model


def pretrain_model_with_residuals(df, n_samples, train_params, target_col, residuals):
    '''
    This function pretrains the model on n samples using the PCA score
    '''
    pretrain_pairs = generate_weighted_pairs(df, n=n_samples, residuals=residuals)
    pretrain_df = create_pair_pca(df, pretrain_pairs, target_col)
    dtrain_pretrain = generate_dmatrix(pretrain_df)
    pretrained_model = xgb.train(train_params, dtrain_pretrain, num_boost_round=XGB_ESTIMATORS)
    return pretrained_model


def pretrain_with_gpt_ranking(df, n_samples, train_params, target_col):
    pretrain_pairs = generate_random_pairs(df, n=n_samples)
    pretrain_df = create_pair_gpt(df, pretrain_pairs, target_col)
    dtrain_pretrain = generate_dmatrix(pretrain_df)
    pretrained_model = xgb.train(train_params, dtrain_pretrain, num_boost_round=XGB_ESTIMATORS)

    return pretrained_model


def generate_all_pairs(df):
    '''
    Generate all possible pairs of indices from the dataframe.
    '''
    pairs = [(i, j) for i in df.index for j in df.index if i != j]
    return pairs


def select_most_uncertain_pairs(model, df, pairs, batch_size, target_col):
    '''
    Select the most uncertain pairs based on the model's predictions.
    Parameters:
        model (xgb.Booster): The current model.
        df (pd.DataFrame): The dataframe containing the data.
        pairs (list of tuples): All possible pairs of indices.
        batch_size (int): The number of uncertain pairs to select.
    Returns:
        list: Selected most uncertain pairs.
    '''
    # Prepare the pair dataframe
    pair_df = create_pair_df(df, pairs, target_col)
    X_pair = pair_df.drop(columns=['label'])
    dpair = xgb.DMatrix(X_pair, enable_categorical=True)
    
    # Get the model's prediction probabilities
    predictions = model.predict(dpair)
    uncertainty = np.abs(predictions - 0.5)  # Uncertainty is highest near 0.5
    
    # Select the indices of the most uncertain pairs
    most_uncertain_indices = np.argsort(uncertainty)[:batch_size]
    selected_pairs = [pairs[i] for i in most_uncertain_indices]
    
    return selected_pairs


def calulate_acc(model, dtest, y_test):
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    return accuracy_score(y_test, y_pred_binary)


def uncertainty_blank(total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, train_params, dtest, y_test):
    current_model_ub = None
    accs = []
    for _ in tqdm(range(0, total_pairs, batch_size), desc="Blank model with uncertainty pairs"):
        if current_model_ub is None:
            sampled_pair_indices = np.random.choice(len(all_pairs), size=batch_size, replace=False)
            selected_pairs = [all_pairs[i] for i in sampled_pair_indices]
        else:
            sampled_pair_indices = np.random.choice(len(all_pairs), size=10_000, replace=False)
            sampled_pairs = [all_pairs[i] for i in sampled_pair_indices]
            selected_pairs = select_most_uncertain_pairs(current_model_ub, df, sampled_pairs, batch_size, target_col)
        
        if add_noise:
            train_df_ub = create_pair_noisy(df, selected_pairs, target_col, variance=noise)
        elif use_bradley:
            if exp:
                train_df_ub = create_pair_bradley_exp(df, selected_pairs, target_col)
            else:
                train_df_ub = create_pair_bradley(df, selected_pairs, target_col)
        else:
            train_df_ub = create_pair_df(df, selected_pairs, target_col)
        
        dtrain_ub = generate_dmatrix(train_df_ub)
        
        if current_model_ub is None:
            current_model_ub = xgb.train(train_params, dtrain_ub, num_boost_round=XGB_ESTIMATORS)
        else:
            current_model_ub = xgb.train(train_params, dtrain_ub, num_boost_round=XGB_ESTIMATORS, xgb_model=current_model_ub)
        
        accs.append(calulate_acc(current_model_ub, dtest, y_test))
    
    return accs


def uncertainty_pretrained(pretrained_model, total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, train_params, dtest, y_test):
    current_model_up = pretrained_model.copy()
    accs = []
    for _ in tqdm(range(0, total_pairs, batch_size), desc="Pretrained model with uncertainty pairs"):
        sampled_pair_indices = np.random.choice(len(all_pairs), size=10_000, replace=False)
        sampled_pairs = [all_pairs[i] for i in sampled_pair_indices]
        selected_pairs = select_most_uncertain_pairs(current_model_up, df, sampled_pairs, batch_size, target_col)
        
        if add_noise:
            train_df_up = create_pair_noisy(df, selected_pairs, target_col, variance=noise)
        elif use_bradley:
            if exp:
                train_df_up = create_pair_bradley_exp(df, selected_pairs, target_col)
            else:
                train_df_up = create_pair_bradley(df, selected_pairs, target_col)
        else:
            train_df_up = create_pair_df(df, selected_pairs, target_col)
        
        dtrain_up = generate_dmatrix(train_df_up)
        current_model_up = xgb.train(train_params, dtrain_up, num_boost_round=XGB_ESTIMATORS, xgb_model=current_model_up)
        
        accs.append(calulate_acc(current_model_up, dtest, y_test))
    
    return accs


def random_blank(total_pairs, batch_size, df, target_col, add_noise, use_bradley, exp, noise, train_params, dtest, y_test):
    accumulated_train_data = []
    accs = []
    for _ in tqdm(range(0, total_pairs, batch_size), desc="Blank model with random pairs"):
        random_pairs = generate_random_pairs(df, n=batch_size)
        if add_noise:
            train_df_rb = create_pair_noisy(df, random_pairs, target_col, variance=noise)
        elif use_bradley:
            if exp:
                train_df_rb = create_pair_bradley_exp(df, random_pairs, target_col)
            else:
                train_df_rb = create_pair_bradley(df, random_pairs, target_col)
        else:
            train_df_rb = create_pair_df(df, random_pairs, target_col)

        accumulated_train_data.append(train_df_rb)
        full_train_df = pd.concat(accumulated_train_data, ignore_index=True)
        dtrain_rb = generate_dmatrix(full_train_df)
        current_model_rb = xgb.train(train_params, dtrain_rb, num_boost_round=XGB_ESTIMATORS)

        accs.append(calulate_acc(current_model_rb, dtest, y_test))

    return accs


def get_n_highest_accs(n, data, test_df, train_params, target_col_name, exp, total_pairs, batch_size):
    acc_list = []
    for itt in range((n)):
        print(f"iteration: {itt+1}")
        acc = calculate_highest_acc(data, test_df, train_params=train_params, target_col=target_col_name, use_bradley=True, exp=exp, total_pairs=total_pairs, batch_size=batch_size)
        acc_list.append(acc)
    return acc_list


def calculate_highest_acc(df, test_df, train_params, target_col, use_bradley, exp, total_pairs, batch_size):
    y_test = test_df['label']
    dtest = generate_dmatrix(test_df)
    all_pairs = generate_all_pairs(df)
    accs = uncertainty_blank(total_pairs, batch_size, all_pairs, df, target_col, False, use_bradley, exp, 0, train_params, dtest, y_test)
    return max(accs)


def compare_three_methods(df, test_df, train_params, pretrained_model, target_col, use_bradley, exp, add_noise, noise, total_pairs, batch_size):
    '''
    Compare three methods:
    1. Blank model with uncertainty-based pairs
    2. Pretrained model with uncertainty-based pairs
    3. Blank model with random pairs
    '''
    
    y_test = test_df['label']
    dtest = generate_dmatrix(test_df)
    all_pairs = generate_all_pairs(df)
    
    acc_UB = uncertainty_blank(total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, train_params, dtest, y_test)
    acc_UP = uncertainty_pretrained(pretrained_model, total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, train_params, dtest, y_test)
    acc_RB = random_blank(total_pairs, batch_size, df, target_col, add_noise, use_bradley, exp, noise, train_params, dtest, y_test)
    
    return acc_UB, acc_UP, acc_RB


def save_accs(filename, ub_scores, up_scores, rb_scores, accs, gpt):
    data = {
        'UB': ub_scores,
        'UP': up_scores,
        'RB': rb_scores,
        'accs': accs,
        'GPT': gpt
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_accs(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['UB'], data['UP'], data['RB'], data['accs'], data['GPT']


def calculate_pca_var(df, target_col_name, useless_cols=[]):
    useless_cols.append(target_col_name)
    features = df.drop(columns=useless_cols)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=1)
    pca_scores = pca.fit_transform(features_scaled)
    reconstructed = pca.inverse_transform(pca_scores)
    residuals = np.linalg.norm(features_scaled - reconstructed, axis=1)
    variance = np.mean(residuals ** 2)
    return variance, residuals


# ============================================================================
# DopeWolfe Algorithm Functions (from tkkiran/DopeWolfe repository)
# ============================================================================


def pretrain_regression_model(data, num_pairs, residuals, target_col):
    # Generate the pretrained pairs
    pretrain_pairs = generate_weighted_pairs(data, n=num_pairs, residuals=residuals)
    pretrain_df = create_pair_pca(data, pretrain_pairs, target_col)
    
    X_pretrain = pretrain_df.drop(columns=['label']).values
    y_pretrain = pretrain_df['label'].values
    
    # Train pretrained model
    pretrained_model_pca = LogisticRegression(
        max_iter=MAX_ITER,
        solver=SOLVER,
        warm_start=True
    )
    pretrained_model_pca.fit(X_pretrain, y_pretrain)
    pretrained_data = (X_pretrain, y_pretrain)
    
    return pretrained_model_pca, pretrained_data


def d_grad(V:  np.ndarray, p: np.ndarray, gamma: float = 1e-6, 
           return_grad: bool = True, subset:  np.ndarray = None) -> Tuple[float, np.ndarray]:
    """
    Value of D-optimal objective and its gradient. 
    
    Parameters: 
        V: n x d x d matrix of feature-vector outer products
        p: distribution over feature vectors (design)
        gamma: regularization parameter
        return_grad: whether to return the gradient
        subset: indices for stochastic gradient computation (DopeWolfe)
    
    Returns:
        obj: objective value (negative log determinant)
        dp: gradient of the objective
    """
    n, d, _ = V.shape
    
    # Inverse of the sample covariance matrix
    G = np.einsum("ijk,i->jk", V, p) + gamma * np.eye(d)
    invG = np.linalg.inv(G)
    
    # Objective value (log det)
    sign, obj = np.linalg.slogdet(G)
    obj *= -sign
    
    if return_grad:
        # Gradient of the objective
        if subset is None:
            M = np.einsum("kl,ilj->ikj", invG, V)
            dp = -np.trace(M, axis1=-2, axis2=-1)
        else:
            M = np.einsum("kl,ilj->ikj", invG, V[subset, :, :])
            dp = np.zeros(n)
            dp[subset] = -np.trace(M, axis1=-2, axis2=-1)
    else:
        dp = 0
    
    return obj, dp


def fw_design(V: np.ndarray, pi_0: np.ndarray = None, R: int = None, 
              num_iters: int = 100, tol: float = 1e-6, printout: bool = False) -> np.ndarray:
    """
    Frank-Wolfe algorithm for D-optimal design optimization.
    
    Parameters:
        V: n x d x d matrix of feature-vector outer products
        pi_0: initial distribution over feature vectors (design)
        R: number of subsampled feature vectors in each iteration (DopeWolfe mode)
        num_iters: maximum number of Frank-Wolfe iterations
        tol: stop when two consecutive objective values differ by less than tol
        printout: whether to print progress
    
    Returns:
        pi: optimized distribution over pairs
    """
    n, d, _ = V.shape
    
    if pi_0 is None: 
        # Initial allocation weights are 1/n and they add up to 1
        pi = np.ones(n) / n
    else:
        pi = np.copy(pi_0)
    
    if R is None:
        R = n
    
    # Frank-Wolfe iterations
    for iter_idx in range(num_iters):
        # Compute gradient at the last solution
        pi_last = np.copy(pi)
        
        if R == n:
            # Dope (full gradient)
            last_obj, grad = d_grad(V, pi_last)
        else:
            # DopeWolfe (stochastic gradient with subsampling)
            last_obj, grad = d_grad(V, pi_last, subset=np.random.permutation(n)[:R])
        
        if iter_idx == 0:
            obj_s = [last_obj]
        
        if printout:
            print(f"{last_obj:.4f}", end=" ")
        
        # Find a feasible LP solution in the direction of the gradient
        pi_lp = np.zeros(n)
        pi_lp[np.argmin(grad)] = 1.0
        
        # Golden-section search in the direction of the gradient
        num_ls_iters = 20
        left_step = 0
        left_obj, _ = d_grad(V, pi_last, return_grad=False)
        right_step = 1.0
        right_obj, _ = d_grad(V, pi_lp, return_grad=False)
        
        for ls_iter in range(num_ls_iters):
            mid1 = left_step + 0.618 * (right_step - left_step)
            obj1, _ = d_grad(V, mid1 * pi_lp + (1 - mid1) * pi_last, return_grad=False)
            mid2 = right_step - 0.618 * (right_step - left_step)
            obj2, _ = d_grad(V, mid2 * pi_lp + (1 - mid2) * pi_last, return_grad=False)
            
            if obj1 < obj2:
                left_step = mid2
                left_obj = obj2
            else:
                right_step = mid1
                right_obj = obj1
        
        best_step = (left_step + right_step) / 2
        
        # Update solution
        pi = best_step * pi_lp + (1 - best_step) * pi_last
        best_obj, _ = d_grad(V, pi, return_grad=False)
        obj_s.append(best_obj)
        
        # Convergence check
        if R == n and last_obj - best_obj < tol:
            break
        elif R != n and len(obj_s) > 2 * (n // R) and obj_s[-(n // R) - 1] - best_obj < tol:
            break
    
    if printout:
        print()
    
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi


def compute_outer_products(df: pd.DataFrame, pairs: List[Tuple[int, int]], 
                           target_col: str) -> np.ndarray:
    """
    Compute outer products V_i = (x_1 - x_2)(x_1 - x_2)^T for each pair.
    
    This is used by the DopeWolfe algorithm to optimize the sampling distribution.
    
    Parameters:
        df: dataframe with features and target
        pairs: list of pair indices
        target_col: name of target column
    
    Returns: 
        V: n_pairs x d x d array of outer products
    """
    df_features = df.drop(columns=[target_col])
    
    # Convert entire dataframe to float array once (more efficient)
    features_array = df_features.values.astype(np.float64)
    
    d = features_array.shape[1]
    n_pairs = len(pairs)
    
    V = np.zeros((n_pairs, d, d))
    
    for i, (idx1, idx2) in enumerate(pairs):
        x1 = features_array[idx1]
        x2 = features_array[idx2]
        diff = x1 - x2
        V[i] = np.outer(diff, diff)
    
    return V


def select_pairs_dopewolfe(df: pd.DataFrame, candidate_pairs: List[Tuple[int, int]], 
                           target_col: str, n_select: int, 
                           use_randomized:  bool = True) -> List[Tuple[int, int]]:
    """
    Use DopeWolfe algorithm to select informative pairs for labeling.
    
    Parameters:
        df: dataframe with features
        candidate_pairs: pool of candidate pairs to select from
        target_col:  name of target column
        n_select: number of pairs to select
        use_randomized: if True, use DopeWolfe (randomized); else use Dope (full)
    
    Returns:
        selected_pairs: list of selected pair indices
    """
    # Compute outer products for all candidate pairs
    V = compute_outer_products(df, candidate_pairs, target_col)
    n_candidates = len(candidate_pairs)
    
    # Run Frank-Wolfe to get optimal distribution
    if use_randomized:
        # DopeWolfe:  use subsampling (R = n/10)
        R = max(n_candidates // 10, 10)
        pi = fw_design(V, R=R, printout=False)
    else:
        # Dope: use full gradient
        pi = fw_design(V, printout=False)
    
    # Sample pairs according to the optimized distribution
    selected_indices = np.random.choice(
        n_candidates, 
        size=min(n_select, n_candidates), 
        replace=False, 
        p=pi
    )
    
    selected_pairs = [candidate_pairs[i] for i in selected_indices]
    return selected_pairs


# ============================================================================
# GURO Algorithm Functions (from HermanBergstrom/GURO repository)
# ============================================================================


class BaseAlgorithm: 
    """Generic algorithm for pairwise comparisons"""
    
    def __init__(self, X, update_every=10, seed=None):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = 0
        self.update_every = update_every
        self.obs_data = []
        self.obs_labels = []
        self.obs_indices = []
        self.random_state = np.random.RandomState(seed)
        self.theta_hat = np.ones((self.d,))
    
    def act(self):
        """Decide which pair to compare next"""
        raise NotImplementedError
    
    def update(self, i, j, observation):
        """Update the algorithm with the observation of the pair (i, j)"""
        self.t += 1
        x = self.X[i] - self.X[j]
        self.obs_data.append(x)
        self.obs_labels.append(observation)
        self.obs_indices.append((i, j))
        if self.t % self.update_every == 0 and self.t >= 10 and len(np.unique(self.obs_labels)) >= 2:
            self.update_model()
    
    def update_model(self):
        """Update the model with the current observations"""
        raise NotImplementedError
    
    def ordering(self):
        """Return ordering of data"""
        scores = self.X.dot(self.theta_hat)
        return rankdata(scores)


class UniformSampling(BaseAlgorithm):
    """Uniform random pair sampling (baseline)"""
    
    def __init__(self, X, update_every=10, seed=None):
        super().__init__(X, update_every, seed)
        self.model = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER)
    
    def act(self):
        return self.random_state.choice(self.n, size=2, replace=False), None
    
    def update_model(self):
        if len(np.unique(self.obs_labels)) < 2:
            return
        self.model.fit(self.obs_data, self.obs_labels)
        self.theta_hat = self.model.coef_[0]


class GURO(BaseAlgorithm):
    """Greedy Uncertainty Reduction for Ordering (GURO)"""
    
    def __init__(self, X, update_every=10, seed=None, sample_combinations=False):
        super().__init__(X, update_every=update_every, seed=seed)
        
        self.sample_combinations = sample_combinations
        
        # Information matrix (starts as identity)
        self.M = np.identity(self.d)
        self.M_inv = np.linalg.inv(self.M)
        
        self.model = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER)
    
    def act(self):
        return self.find_best_pair(sample=self.sample_combinations)
    
    def find_best_pair(self, sample=False, combinations=None):
        """Find the pair that maximizes uncertainty reduction"""
        if combinations is None: 
            combinations = np.array(list(itertools.combinations(range(self.n), 2)))
        
        # Subsample if too many combinations
        if sample and len(combinations) > 5000:
            sample_idxs = self.random_state.choice(len(combinations), 5000, replace=False)
            combinations = combinations[sample_idxs]
        
        # x_i - x_j for all combinations
        diff_matrix = np.array([self.X[c[0]] - self.X[c[1]] for c in combinations])
        
        # Compute exploration term (uncertainty) - select pair with highest uncertainty
        exploration_term = np.sqrt(np.sum(diff_matrix.dot(self.M_inv) * diff_matrix, axis=1))
        
        idx = np.argmax(exploration_term)
        return combinations[idx], None
    
    def update(self, i, j, observation):
        """Update algorithm with observation"""
        super().update(i, j, observation)
        
        # Update information matrix
        x = self.X[i] - self.X[j]
        y_hat = expit(x.dot(self.theta_hat))
        coef = y_hat * (1 - y_hat)
        self.M = self.M + coef * np.outer(x, x)
        self.M_inv = np.linalg.inv(self.M)
    
    def update_model(self):
        if len(np.unique(self.obs_labels)) < 2:
            return
        self.model.fit(self.obs_data, self.obs_labels)
        self.theta_hat = self.model.coef_[0]


class GURORealData(GURO):
    """GURO with restricted available comparisons (for real datasets)"""
    
    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = np.array(available_combinations)
    
    def act(self):
        # Random sampling for first 10 steps to get initial model
        if self.t < 10:
            index = self.random_state.choice(len(self.available_combinations), size=1, replace=False)
            return self.available_combinations[index[0]], None
        
        return self.find_best_pair(sample=self.sample_combinations, combinations=self.available_combinations)
    
    def update(self, i, j, observation):
        """Update and remove used pair from available combinations"""
        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(self.available_combinations, index, 0)
                break
        super().update(i, j, observation)


class UniformSamplingRealData(UniformSampling):
    """Uniform sampling with restricted available comparisons"""
    
    def __init__(self, *args, available_combinations, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_combinations = np.array(available_combinations)
    
    def act(self):
        index = self.random_state.choice(len(self.available_combinations), size=1, replace=False)
        return self.available_combinations[index[0]], None
    
    def update(self, i, j, observation):
        """Update and remove used pair from available combinations"""
        for index in range(len(self.available_combinations)):
            x1, x2 = self.available_combinations[index]
            if (x1 == i and x2 == j) or (x2 == i and x1 == j):
                self.available_combinations = np.delete(self.available_combinations, index, 0)
                break
        super().update(i, j, observation)


