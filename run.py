import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import Config.util as util_module
from Config.util import (
	add_gpt_rank,
	calulate_acc,
	calculate_highest_acc,
	calculate_pca_var,
	compare_three_methods,
	create_pair_df,
	generate_dmatrix,
	generate_random_pairs,
	pretrain_model_with_residuals,
	pretrain_with_gpt_ranking,
	save_accs,
	standardize_features,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Train and evaluate the active learning framework used in the project notebooks. "
			"Provide a cleaned dataset along with optional GPT rankings to reproduce the experiments."
		)
	)
	parser.add_argument(
		"--dataset-path",
		type=Path,
		default=Path("Datasets/final_data.csv"),
		help="Path to the cleaned CSV dataset." \
			 " Defaults to the FIFA processed file.",
	)
	parser.add_argument(
		"--target-col",
		type=str,
		default="value_eur",
		help="Name of the target column used for ranking comparisons.",
	)
	parser.add_argument(
		"--gpt-ranking-path",
		type=Path,
		default=None,
		help="CSV with GPT benchmark rankings. Required for GPT pretraining if not already joined.",
	)
	parser.add_argument(
		"--no-standardize",
		action="store_true",
		help="Disable feature standardization before training.",
	)
	parser.add_argument(
		"--pca-components",
		type=int,
		default=2,
		help="Number of PCA components to compute. The first component is used for PCA supervision.",
	)
	parser.add_argument(
		"--num-test-pairs",
		type=int,
		default=20000,
		help="Number of random test pairs used for evaluation.",
	)
	parser.add_argument(
		"--total-pairs",
		type=int,
		default=800,
		help="Total number of training pairs gathered during each active learning run.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=50,
		help="Batch size for each acquisition step inside the active learning loops.",
	)
	parser.add_argument(
		"--repeats",
		type=int,
		default=10,
		help="How many times to repeat the compare_three_methods experiment for averaging.",
	)
	parser.add_argument(
		"--alpha-coef",
		type=float,
		default=1e-6,
		help="Alpha factor used in the heuristic num_pairs calculation.",
	)
	parser.add_argument(
		"--max-pairs",
		type=int,
		default=-1,
		help="Hard cap for the num_pairs heuristic. If negative, len(dataset) * multiplier is used.",
	)
	parser.add_argument(
		"--max-pairs-multiplier",
		type=float,
		default=1.0,
		help="Multiplier applied to dataset length when max-pairs is not set.",
	)
	parser.add_argument(
		"--max-depth",
		type=int,
		default=0,
		help="Override for XGBoost tree depth. Defaults to sqrt(#features).",
	)
	parser.add_argument(
		"--eta",
		type=float,
		default=0.3,
		help="Learning rate for XGBoost training.",
	)
	parser.add_argument(
		"--subsample",
		type=float,
		default=0.8,
		help="Subsample ratio used by XGBoost.",
	)
	parser.add_argument(
		"--colsample-bytree",
		type=float,
		default=0.8,
		help="Column subsample ratio for XGBoost.",
	)
	parser.add_argument(
		"--lambda-reg",
		type=float,
		default=1.0,
		help="L2 regularization term for XGBoost (lambda).",
	)
	parser.add_argument(
		"--gamma",
		type=float,
		default=0.0,
		help="Minimum loss reduction required to make a further partition on a leaf node of the tree.",
	)
	parser.add_argument(
		"--num-estimators",
		type=int,
		default=500,
		help="Number of boosting estimators used for every training call in util.py.",
	)
	parser.add_argument(
		"--use-bradley",
		action="store_true",
		help="Toggle Bradley-Terry style synthetic labeling inside active learning routines.",
	)
	parser.add_argument(
		"--exp-bradley",
		action="store_true",
		help="Use the exponential Bradley-Terry variant whenever --use-bradley is active.",
	)
	parser.add_argument(
		"--add-noise",
		action="store_true",
		help="Inject +/- noise into labels when building pairs.",
	)
	parser.add_argument(
		"--noise-level",
		type=float,
		default=0.05,
		help="Relative variance used for noisy label generation (only if --add-noise).",
	)
	parser.add_argument(
		"--skip-active-learning",
		action="store_true",
		help="Skip the compare_three_methods loop to only report pretrained accuracies.",
	)
	parser.add_argument(
		"--skip-highest-acc",
		action="store_true",
		help="Avoid running the expensive calculate_highest_acc benchmark.",
	)
	parser.add_argument(
		"--skip-gpt-pretraining",
		action="store_true",
		help="Skip GPT-based pretraining even if GPT rankings are available.",
	)
	parser.add_argument(
		"--output-accs",
		type=Path,
		default=None,
		help="Optional path for persisting accuracy curves via util.save_accs().",
	)
	parser.add_argument(
		"--summary-json",
		type=Path,
		default=None,
		help="Optional path for writing a small JSON summary with the main metrics.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for NumPy and pandas shuffling routines.",
	)
	return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
	if not args.dataset_path.exists():
		raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
	if args.gpt_ranking_path and not args.gpt_ranking_path.exists():
		raise FileNotFoundError(f"GPT ranking file not found: {args.gpt_ranking_path}")


def attach_pca_feature(df: pd.DataFrame, target_col: str, components: int) -> pd.DataFrame:
	drop_cols = {target_col}
	if 'GPT' in df.columns:
		drop_cols.add('GPT')
	feature_cols = [col for col in df.columns if col not in drop_cols]
	if not feature_cols:
		raise ValueError("No feature columns available for PCA computation.")
	pca = PCA(n_components=components)
	transformed = pca.fit_transform(df[feature_cols])
	df['PCA'] = transformed[:, 0]
	return df


def infer_tree_depth(num_features: int, override: int) -> int:
	if override and override > 0:
		return override
	inferred = int(round(np.sqrt(max(1, num_features))))
	return inferred


def compute_num_pairs(num_rows: int, args: argparse.Namespace, variance: float) -> int:
	if args.max_pairs > 0:
		max_pairs = args.max_pairs
	else:
		max_pairs = int(num_rows * args.max_pairs_multiplier)
	max_pairs = max(2, max_pairs)
	num_pairs = int(max_pairs / (1 + args.alpha_coef * variance))
	return max(2, num_pairs)


def prepare_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
	data = pd.read_csv(args.dataset_path)
	if args.gpt_ranking_path:
		data = add_gpt_rank(data, str(args.gpt_ranking_path))
	if args.target_col not in data.columns:
		raise KeyError(f"Target column '{args.target_col}' not present in the dataset.")
	if not args.no_standardize:
		data = standardize_features(data, args.target_col)
	data = attach_pca_feature(data, args.target_col, args.pca_components)
	data_with_gpt = None
	if 'GPT' in data.columns:
		data_with_gpt = data.copy().reset_index(drop=True)
		data = data.drop(columns=['GPT']).reset_index(drop=True)
	else:
		data = data.reset_index(drop=True)
	return data, data_with_gpt


def build_train_params(args: argparse.Namespace, num_features: int) -> Dict[str, float]:
	depth = infer_tree_depth(num_features, args.max_depth)
	params = {
		'objective': 'binary:logistic',
		'eval_metric': 'logloss',
		'max_depth': depth,
		'eta': args.eta,
		'subsample': args.subsample,
		'colsample_bytree': args.colsample_bytree,
		'lambda': args.lambda_reg,
		'gamma': args.gamma,
	}
	return params


def summarize_active_learning(accs: List[List[float]]) -> Dict[str, float]:
	if not accs:
		return {}
	arr = np.array(accs)
	return {
		'mean_curve': arr.mean(axis=0).tolist(),
		'std_curve': arr.std(axis=0).tolist(),
		'final_mean': float(arr.mean(axis=0)[-1]),
	}


def main() -> None:
	args = parse_args()
	validate_paths(args)
	np.random.seed(args.seed)

	util_module.XGB_ESTIMATORS = args.num_estimators

	data, data_with_gpt = prepare_data(args)

	useless_cols = []
	if 'PCA' in data.columns:
		useless_cols.append('PCA')
	variance, residuals = calculate_pca_var(data, args.target_col, useless_cols)
	num_pairs = compute_num_pairs(len(data), args, variance)

	test_pairs = generate_random_pairs(data, args.num_test_pairs)
	test_df = create_pair_df(data, test_pairs, args.target_col)
	y_test = test_df['label']
	dtest = generate_dmatrix(test_df)

	train_params = build_train_params(args, num_features=len(data.columns) - 1)

	pretrained_model_pca = pretrain_model_with_residuals(
		df=data,
		n_samples=num_pairs,
		train_params=train_params,
		target_col=args.target_col,
		residuals=residuals,
	)

	pretrained_model_gpt = None
	if data_with_gpt is not None and not args.skip_gpt_pretraining:
		pretrained_model_gpt = pretrain_with_gpt_ranking(
			df=data_with_gpt,
			n_samples=num_pairs,
			train_params=train_params,
			target_col=args.target_col,
		)

	pca_acc = calulate_acc(pretrained_model_pca, dtest, y_test)
	gpt_acc = None
	if pretrained_model_gpt is not None:
		gpt_acc = calulate_acc(pretrained_model_gpt, dtest, y_test)

	highest_acc = None
	if not args.skip_highest_acc:
		highest_acc = calculate_highest_acc(
			data,
			test_df,
			train_params=train_params,
			target_col=args.target_col,
			use_bradley=args.use_bradley,
			exp=args.exp_bradley,
			total_pairs=args.total_pairs,
			batch_size=args.batch_size,
		)

	accs_ub_runs: List[List[float]] = []
	accs_up_runs: List[List[float]] = []
	accs_rb_runs: List[List[float]] = []
	if not args.skip_active_learning:
		for repeat in range(args.repeats):
			print(f"Active learning repeat {repeat + 1}/{args.repeats}")
			acc_ub, acc_up, acc_rb = compare_three_methods(
				df=data,
				test_df=test_df,
				train_params=train_params,
				pretrained_model=pretrained_model_pca,
				target_col=args.target_col,
				use_bradley=args.use_bradley,
				exp=args.exp_bradley,
				add_noise=args.add_noise,
				noise=args.noise_level,
				total_pairs=args.total_pairs,
				batch_size=args.batch_size,
			)
			accs_ub_runs.append(acc_ub)
			accs_up_runs.append(acc_up)
			accs_rb_runs.append(acc_rb)

	summary = {
		'dataset': str(args.dataset_path),
		'target_column': args.target_col,
		'num_rows': len(data),
		'num_pairs': num_pairs,
		'variance': variance,
		'pca_accuracy': pca_acc,
		'gpt_accuracy': gpt_acc,
		'highest_acc': highest_acc,
		'active_learning': {
			'UB': summarize_active_learning(accs_ub_runs),
			'UP': summarize_active_learning(accs_up_runs),
			'RB': summarize_active_learning(accs_rb_runs),
		},
	}

	print("\n=== Summary ===")
	print(json.dumps({k: v for k, v in summary.items() if k != 'active_learning'}, indent=2))
	if summary['active_learning']['UB']:
		print("Final averaged UB accuracy:", summary['active_learning']['UB']['final_mean'])
	if summary['active_learning']['UP']:
		print("Final averaged UP accuracy:", summary['active_learning']['UP']['final_mean'])
	if summary['active_learning']['RB']:
		print("Final averaged RB accuracy:", summary['active_learning']['RB']['final_mean'])

	if args.output_accs and not args.skip_active_learning:
		args.output_accs.parent.mkdir(parents=True, exist_ok=True)
		save_accs(
			str(args.output_accs),
			accs_ub_runs,
			accs_up_runs,
			accs_rb_runs,
			highest_acc,
			gpt_acc,
		)
	if args.summary_json:
		args.summary_json.parent.mkdir(parents=True, exist_ok=True)
		with open(args.summary_json, 'w', encoding='utf-8') as fp:
			json.dump(summary, fp, indent=2)


if __name__ == "__main__":
	main()

