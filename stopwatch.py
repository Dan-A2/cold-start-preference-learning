from Config.util import *
import time


# Load and preprocess the dataset
data = pd.read_csv('Datasets/Student_performance_data _.csv')
target_col_name = 'GPA'
data.drop(columns=['GradeClass', 'StudentID'], inplace=True)
object_columns = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring',
                'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
numeric_columns = ['Age', 'StudyTimeWeekly', 'Absences']
object_columns.remove('Tutoring')
numeric_columns.append('Tutoring')
data[numeric_columns] = data[numeric_columns].astype(int)
data[object_columns] = data[object_columns].astype(str)
data = pd.get_dummies(data, columns=object_columns)
data = standardize_features(data, target_col_name)

# PCA + Residual + Pretrain (our method)
start = time.time()
X = data.drop(columns=[target_col_name])

pca = PCA(n_components=2)
pca.fit(X)
data['PCA'] = pca.transform(X)[:, 0]
tree_depth = int(np.round(np.sqrt(len(data.columns))))
var, residuals = calculate_pca_var(data, target_col_name)
max_pairs = len(data) * 10
alpha = 1e-6

num_pairs = int(max_pairs / (1 + alpha * var))
train_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': tree_depth
}
pretrained_model_pca = pretrain_model_with_residuals(
    df=data,
    n_samples=num_pairs,
    train_params=train_params,
    target_col=target_col_name,
    residuals=residuals
)
total = time.time() - start

print(f"{total:.3f} seconds")