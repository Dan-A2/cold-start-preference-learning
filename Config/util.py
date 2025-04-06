import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb
from tqdm import tqdm


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


def find_highest_cosine_distance(df, label_column):
    """
    This function finds the two columns with the highest cosine distance
    and the corresponding value that caused it.
    """
    # Initialize variables to keep track of the results
    highest_distance = 0
    col = None
    causing_value = None
    
    second_highest_distance = 0
    second_col = None
    second_causing_value = None

    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if label_column in object_cols:
        object_cols.remove(label_column)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_column in numeric_cols:
        numeric_cols.remove(label_column)

    scaler = StandardScaler()

    # Loop through all combinations of categorical columns
    for target_column in object_cols:
        # Extract unique values from the target column
        unique_values = df[target_column].unique()

        for value in unique_values:
            subset_without_value = df[df[target_column] != value][numeric_cols]
            subset_with_value = df[df[target_column] == value][numeric_cols]
            subset_without_value_scaled = scaler.fit_transform(subset_without_value)
            subset_with_value_scaled = scaler.fit_transform(subset_with_value)
            pca_without_value = PCA(n_components=1)
            pca_with_value = PCA(n_components=1)
            pca_without_value.fit(subset_without_value_scaled)
            pca_with_value.fit(subset_with_value_scaled)
            eigenvector_without_value = pca_without_value.components_[0]
            eigenvector_with_value = pca_with_value.components_[0]
            cosine_similarity = np.dot(eigenvector_without_value, eigenvector_with_value) / (
                np.linalg.norm(eigenvector_without_value) * np.linalg.norm(eigenvector_with_value)
            )
            cosine_distance = 1 - cosine_similarity
            
            if cosine_distance > highest_distance:
                highest_distance = cosine_distance
                col = target_column
                causing_value = value
            
            elif highest_distance > cosine_distance > second_highest_distance and col != None and target_column != col[0]:
                second_highest_distance = cosine_distance
                second_col = target_column
                second_causing_value = value

    return {
        'columns': (col, second_col),
        'highest_cosine_distance': (highest_distance, second_highest_distance),
        'causing_value': (causing_value, second_causing_value)
    }


def drop_redundant_cols(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)


def divide_dataframe_by_causing_values(df, result):
    """
    Divides the dataframe into 4 parts based on causing values of the highest and second-highest cosine distances.

    Parameters:
    - df: Original dataframe
    - result: Output from find_highest_cosine_distance function

    Returns:
    - A dictionary containing the 4 dataframes
    """
    col1, col2 = result['columns']
    value1, value2 = result['causing_value']
    col1 = col1 + "_" + value1
    col2 = col2 + "_" + value2

    mask1 = df[col1] == True
    mask2 = df[col2] == True

    df_1 = df[mask1 & mask2]
    df_2 = df[mask1 & ~mask2]
    df_3 = df[~mask1 & mask2]
    df_4 = df[~mask1 & ~mask2]

    return df_1, df_2, df_3, df_4
    


def combine_pca_mca(pca_values, mca_values, pca_weight=0.5, mca_weight=0.5):
    '''
    The name is self-explanatory
    '''
    pca_normalized = (pca_values - np.min(pca_values)) / (np.max(pca_values) - np.min(pca_values))
    mca_normalized = (mca_values - np.min(mca_values)) / (np.max(mca_values) - np.min(mca_values))
    combined_scores = (pca_weight * pca_normalized) + (mca_weight * mca_normalized)
    return combined_scores


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


def train_evaluate_repeat(generate_pairs_func, df, num_samples, test_df, label_col, use_bradley, exp, add_noise, noise=0.05, depth=5, repeats=10):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for _ in tqdm(range(repeats)):
        # Generate pairs and create the training dataframe
        pairs = generate_pairs_func(df, num_samples)
        
        if add_noise:
            train_df = create_pair_noisy(df, pairs, label_col, variance=noise)
        elif use_bradley:
            if exp:
                train_df = create_pair_bradley_exp(df, pairs, label_col)
            else:
                train_df = create_pair_bradley(df, pairs, label_col)
        else:
            train_df = create_pair_df(df, pairs, label_col)
        
        # Prepare the training data
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        
        # Prepare the test data
        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']
        
        # Train the model
        model = xgb.XGBClassifier(max_depth=depth, n_estimators=XGB_ESTIMATORS, eval_metric='logloss', enable_categorical=True)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy, precision, recall, _, f1 = calculate_metrics(y_test, y_pred)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")
    return mean_f1


def calculate_metrics(y_test, y_pred):
    '''
    This function calculates the required metrics to assess model performance
    '''
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, conf_matrix, f1


def pretrain_model(df, n_samples, pretrain_params, target_col):
    '''
    This function pretrains the model on n samples using the PCA score
    '''
    pretrain_pairs = generate_random_pairs(df, n=n_samples)
    pretrain_df = create_pair_pca(df, pretrain_pairs, target_col)
    X_pretrain = pretrain_df.drop(columns=['label'])
    y_pretrain = pretrain_df['label']
    dtrain_pretrain = xgb.DMatrix(X_pretrain, label=y_pretrain, enable_categorical=True)
    pretrained_model = xgb.train(pretrain_params, dtrain_pretrain, num_boost_round=XGB_ESTIMATORS)
    return pretrained_model


def pretrain_model_with_residuals(df, n_samples, pretrain_params, target_col, residuals):
    '''
    This function pretrains the model on n samples using the PCA score
    '''
    pretrain_pairs = generate_weighted_pairs(df, n=n_samples, residuals=residuals)
    pretrain_df = create_pair_pca(df, pretrain_pairs, target_col)
    X_pretrain = pretrain_df.drop(columns=['label'])
    y_pretrain = pretrain_df['label']
    dtrain_pretrain = xgb.DMatrix(X_pretrain, label=y_pretrain, enable_categorical=True)
    pretrained_model = xgb.train(pretrain_params, dtrain_pretrain, num_boost_round=XGB_ESTIMATORS)
    return pretrained_model


def pretrain_model_with_residuals_multi(dfs, n_samples, pretrain_params, target_col, residual_list, total_len):
    """
    Pretrains the model on the concatenation of all given dataframes using residuals.
    
    Parameters:
    - dfs: List of dataframes to use for pretraining.
    - n_samples: Number of samples to generate for training.
    - pretrain_params: XGBoost training parameters.
    - target_col: Target column name.
    - residual_list: List of residuals corresponding to each dataframe.

    Returns:
    - pretrained XGBoost model.
    """
    all_pretrain_dfs = []
    for i in range(len(dfs)):
        df = dfs[i]
        residuals = residual_list[i]
        df.reset_index(drop=True, inplace=True)
        pretrain_pairs = generate_weighted_pairs(df, n=(n_samples * len(df)) // total_len, residuals=residuals)
        pretrain_df = create_pair_pca(df, pretrain_pairs, target_col)     
        all_pretrain_dfs.append(pretrain_df)
    final_pretrain_df = pd.concat(all_pretrain_dfs, ignore_index=True)
    X_pretrain = final_pretrain_df.drop(columns=['label'])
    y_pretrain = final_pretrain_df['label']
    dtrain_pretrain = xgb.DMatrix(X_pretrain, label=y_pretrain, enable_categorical=True)
    pretrained_model = xgb.train(pretrain_params, dtrain_pretrain, num_boost_round=XGB_ESTIMATORS)
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


def compare_finetuned_blank(df, test_df, pretrain_params, pretrained_model, target_col):
    '''
    This function compares the fine-tuned model with the blank model
    '''
    # Initialize lists to store F1 scores
    f1_scores_pretrained = []
    f1_scores_untrained = []

    # Prepare the test data
    X_test = test_df.drop(columns=[f'{target_col}_1', f'{target_col}_2', 'label'])
    y_test = test_df['label']
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    for n in range(10, 501, 10):
        # Generate n random pairs for fine-tuning
        pairs_fine_tune = generate_random_pairs(df, n=n)
        train_df_fine_tune = create_pair_pca(df, pairs_fine_tune, target_col)
        
        # Prepare the fine-tuning training data
        X_train_fine_tune = train_df_fine_tune.drop(columns=['label'])
        y_train_fine_tune = train_df_fine_tune['label']
        dtrain_fine_tune = xgb.DMatrix(X_train_fine_tune, label=y_train_fine_tune, enable_categorical=True)
        
        # Fine-tune the pretrained model
        finetuned_model = xgb.train(pretrain_params, dtrain_fine_tune, num_boost_round=XGB_ESTIMATORS, xgb_model=pretrained_model)
        y_pred_finetuned = finetuned_model.predict(dtest)
        y_pred_finetuned_binary = (y_pred_finetuned > 0.5).astype(int)
        f1_finetuned = f1_score(y_test, y_pred_finetuned_binary)
        f1_scores_pretrained.append(f1_finetuned)
        
        # Train the blank model from scratch on the same data
        untrained_model = xgb.train(pretrain_params, dtrain_fine_tune, num_boost_round=XGB_ESTIMATORS)
        y_pred_untrained = untrained_model.predict(dtest)
        y_pred_untrained_binary = (y_pred_untrained > 0.5).astype(int)
        f1_untrained = f1_score(y_test, y_pred_untrained_binary)
        f1_scores_untrained.append(f1_untrained)
    
    return f1_scores_pretrained, f1_scores_untrained


def uncertainty_blank(total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, pretrain_params, dtest, y_test):
    current_model_ub = None
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
        
        X_train_ub = train_df_ub.drop(columns=['label'])
        y_train_ub = train_df_ub['label']
        dtrain_ub = xgb.DMatrix(X_train_ub, label=y_train_ub, enable_categorical=True)
        
        if current_model_ub is None:
            current_model_ub = xgb.train(pretrain_params, dtrain_ub, num_boost_round=XGB_ESTIMATORS)
        else:
            current_model_ub = xgb.train(pretrain_params, dtrain_ub, num_boost_round=XGB_ESTIMATORS, xgb_model=current_model_ub)
        
        y_pred_ub = current_model_ub.predict(dtest)
        y_pred_ub_binary = (y_pred_ub > 0.5).astype(int)
        return f1_score(y_test, y_pred_ub_binary)


def uncertainty_pretrained(pretrained_model, total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, pretrain_params, dtest, y_test):
    current_model_up = pretrained_model.copy()
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
        
        X_train_up = train_df_up.drop(columns=['label'])
        y_train_up = train_df_up['label']
        dtrain_up = xgb.DMatrix(X_train_up, label=y_train_up, enable_categorical=True)
        
        current_model_up = xgb.train(pretrain_params, dtrain_up, num_boost_round=XGB_ESTIMATORS, xgb_model=current_model_up)
        y_pred_up = current_model_up.predict(dtest)
        y_pred_up_binary = (y_pred_up > 0.5).astype(int)
        return f1_score(y_test, y_pred_up_binary)


def random_blank(total_pairs, batch_size, df, target_col, add_noise, use_bradley, exp, noise, pretrain_params, dtest, y_test):
    current_model_rb = None
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
        
        X_train_rb = train_df_rb.drop(columns=['label'])
        y_train_rb = train_df_rb['label']
        dtrain_rb = xgb.DMatrix(X_train_rb, label=y_train_rb, enable_categorical=True)
        
        if current_model_rb is None:
            current_model_rb = xgb.train(pretrain_params, dtrain_rb, num_boost_round=XGB_ESTIMATORS)
        else:
            current_model_rb = xgb.train(pretrain_params, dtrain_rb, num_boost_round=XGB_ESTIMATORS, xgb_model=current_model_rb)
        
        y_pred_rb = current_model_rb.predict(dtest)
        y_pred_rb_binary = (y_pred_rb > 0.5).astype(int)
        return f1_score(y_test, y_pred_rb_binary)


def compare_three_methods(df, test_df, pretrain_params, pretrained_model, target_col, use_bradley, exp, add_noise, noise=0.05, total_pairs=100, batch_size=10):
    '''
    Compare three methods:
    1. Blank model with uncertainty-based pairs
    2. Pretrained model with uncertainty-based pairs
    3. Blank model with random pairs
    '''
    f1_scores_UB = []
    f1_scores_UP = []
    f1_scores_RB = []

    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    all_pairs = generate_all_pairs(df)
    
    f1_ub = uncertainty_blank(total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, pretrain_params, dtest, y_test)
    f1_scores_UB.append(f1_ub)
    f1_up = uncertainty_pretrained(pretrained_model, total_pairs, batch_size, all_pairs, df, target_col, add_noise, use_bradley, exp, noise, pretrain_params, dtest, y_test)
    f1_scores_UP.append(f1_up)
    f1_rb = random_blank(total_pairs, batch_size, df, target_col, add_noise, use_bradley, exp, noise, pretrain_params, dtest, y_test)
    f1_scores_RB.append(f1_rb)

    return f1_scores_UB, f1_scores_UP, f1_scores_RB


def calculate_pca_var(df, target_col_name, useless_cols=[]):
    useless_cols.append(target_col_name)
    features = df.drop(columns=useless_cols)
    target = df[target_col_name]
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(features)

    # Project the data points back to the PCA line
    # pca_result[:, 0] gives the scalar projections (1-D array)
    aligned_pca_values = pca_result[:, 0]  # These are the 1D PCA projections

    # Compute residuals between the actual target values and the PCA projections
    # The PCA projections are scaled, so align them with the target
    residuals = target - aligned_pca_values

    # Calculate variance from residuals
    variance = np.mean(residuals**2)
    return variance, residuals


def calculate_pca_var_multi(dfs, target_col_name, useless_cols=[]):
    total_variance = 0
    all_residuals = []

    for df in dfs:
        var, res = calculate_pca_var(df, target_col_name, useless_cols)
        total_variance += var
        all_residuals.append(res)

    return total_variance, all_residuals  # all_residuals is now a 1D list
