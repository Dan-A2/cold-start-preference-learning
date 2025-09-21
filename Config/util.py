import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tqdm import tqdm
import pickle


XGB_ESTIMATORS = 500


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


def save_accs(filename, ub_scores, up_scores, rb_scores, line, gpt):
    data = {
        'UB': ub_scores,
        'UP': up_scores,
        'RB': rb_scores,
        'line': line,
        'GPT': gpt
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_accs(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['UB'], data['UP'], data['RB'], data['line'], data['GPT']


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
