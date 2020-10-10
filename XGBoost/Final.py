import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
cols = ['u', 'g', 'r', 'i', 'z', 'redshift', 'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 'nObserve', 'nDetect', 'airmass_u', 'airmass_g', 'airmass_r', 'airmass_i', 'airmass_z']
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
x = train.drop(columns='class', axis=1)
y = train['class']  # 결과 레이블(class)
TEST = test
# pd.set_option('display.max_rows', 1000)
# # 최대 열 수 설정
# pd.set_option('display.max_columns', 1000)
# # 표시할 가로의 길이
# pd.set_option('display.width', 1000)
# # 파라미터 후보
# param_grid = {
#                  'n_jobs': [-1],
#                  'max_depth': list(range(7, 14)),
#                  'colsample_bytree': list(np.arange(0.6, 1.05, 0.1)),
#                  'subsample' : list(np.arange(0.6, 1.05, 0.1)),
#                  'learning_rate' : list(np.arange(0.01, 0.15, 0.02)),
#                  'n_estimators':[2000],
#                  'objective':["multi:softmax"],
#                  'random_state':[42]
#             }

df = pd.concat([x, TEST], axis=0)
df.fillna(-1, inplace=True)
d1 = ['u', 'g', 'r', 'i', 'z']
d2 = ['dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z']
am = ['airmass_u', 'airmass_g', 'airmass_r', 'airmass_i', 'airmass_z']

# 통계값을 이용
dered = df[['dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z']]
df['dered_mean'] = dered.mean(axis=1)
df['dered_var'] = dered.var(axis=1)
df['dered_std'] = dered.std(axis=1)
df['dered_sum'] = dered.sum(axis=1)
df['dered_median'] = dered.median(axis=1)
df['d_dered_u'] = (df['dered_u'] - df['u'])
df['d_dered_g'] = (df['dered_g'] - df['g'])
df['d_dered_r'] = (df['dered_r'] - df['r'])
df['d_dered_i'] = (df['dered_i'] - df['i'])
df['d_dered_z'] = (df['dered_z'] - df['z'])
df['d_dered_ug'] = (df['dered_u'] - df['dered_g'])
df['d_dered_gr'] = (df['dered_g'] - df['dered_r'])
df['d_dered_gi'] = (df['dered_g'] - df['dered_i'])
df['d_dered_gz'] = (df['dered_g'] - df['dered_z'])
df['d_dered_ri'] = (df['dered_r'] - df['dered_i'])
df['d_dered_rz'] = (df['dered_r'] - df['dered_z'])
df['d_dered_iz'] = (df['dered_i'] - df['dered_z'])
df['d_obs_det'] = (df['nObserve'] - df['nDetect'])
df['d_dered_u2'] = (df['dered_u'] + df['u'])
df['d_dered_g2'] = (df['dered_g'] + df['g'])
df['d_dered_r2'] = (df['dered_r'] + df['r'])
df['d_dered_z2'] = (df['dered_z'] + df['z'])
df['d_dered_i2'] = (df['dered_i'] + df['i'])
df['d_dered_ug2'] = (df['dered_u'] + df['dered_g'])
df['d_dered_gr2'] = (df['dered_g'] + df['dered_r'])
df['d_dered_gi2'] = (df['dered_g'] + df['dered_i'])
df['d_dered_gz2'] = (df['dered_g'] + df['dered_z'])
df['d_dered_ri2'] = (df['dered_r'] + df['dered_i'])
df['d_dered_rz2'] = (df['dered_r'] + df['dered_z'])
df['d_dered_iz2'] = (df['dered_i'] + df['dered_z'])
df['ug'] = df['u'] * df['g']
df['gr'] = df['g'] * df['r']
df['ri'] = df['r'] * df['i']
df['iz'] = df['i'] * df['z']
df['gi'] = df['g'] * df['i']
df['c1'] = (df['r'] - df['i']) - (df['g'] - df['r']) / 4 - 0.177
df['c2'] = 0.7 * (df['g'] - df['r']) + 1.2 * ((df['r'] - df['i']) - 0.177)
df['l-color'] = -0.436 * df['u'] + 1.129 * df['g'] - 0.119 * df['r'] - 0.574 * df['i'] + 0.1984
df['s-color'] = -0.249 * df['u'] + 0.794 * df['g'] - 0.555 * df['r'] + 0.234
df['P1'] = 0.91 * df['u'] * df['g'] + 0.415 * df['g'] * df['r'] - 1.280

df = df[['u', 'g', 'r', 'i', 'z', 'redshift', 'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 'nObserve', 'nDetect', 'airmass_u', 'airmass_g', 'airmass_r', 'airmass_i', 'airmass_z', 'dered_mean', 'dered_var', 'dered_std', 'dered_median', 'd_dered_u', 'd_dered_g', 'd_dered_z', 'd_dered_ug', 'd_dered_gr', 'd_dered_gi', 'd_dered_gz', 'd_dered_ri', 'd_dered_rz', 'd_dered_iz', 'd_obs_det', 'd_dered_u2', 'd_dered_g2', 'd_dered_r2', 'd_dered_i2', 'd_dered_z2', 'd_dered_ug2', 'd_dered_gr2', 'd_dered_gi2', 'd_dered_gz2', 'd_dered_ri2', 'd_dered_rz2', 'd_dered_iz2', 'ug', 'gr', 'ri', 'iz', 'gi', 'c1', 'c2', 'l-color', 's-color', 'P1']]
c = df.columns
x = df[:320000]
print(x.shape)
TEST = df[320000:]
scaler = StandardScaler()
x = scaler.fit_transform(x)
TEST = scaler.transform(TEST)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ans = np.zeros([80000, 3])
select = np.zeros(len(c))
score = []

# model = XGBClassifier(tree_method='gpu_hist', n_jobs=-1)
# gcv = GridSearchCV(model, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
# gcv.fit(x, y)
# print('best params', gcv.best_params_)   # 최적의 파라미터 값
# print('best score', gcv.best_score_)    # 최고의 점수

for train_idx, test_idx in skf.split(x, y):
    train_x, test_x = x[train_idx], x[test_idx]
    train_y, test_y = y[train_idx], y[test_idx]
    evals = [(test_x, test_y)]
    xgb = XGBClassifier(tree_method='gpu_hist', n_estimators=2000, n_jobs=-1, learning_rate=0.05, subsample=0.65, max_depth=50, objective="multi:softmax", random_state=42)
    xgb.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
    print("train acc : {}".format(xgb.score(train_x, train_y)))
    print("test acc : {}".format(xgb.score(test_x, test_y)))
    print(classification_report(test_y, xgb.predict(test_x), target_names=['class 0', 'class 1', 'class 2']))
    # results = permutation_importance(xgb, test_x, test_y, n_jobs=-1, n_repeats=1, scoring='accuracy')
    # importance = results.importances
    # importance = importance.flatten()
    # select += importance
    score.append(xgb.score(test_x, test_y))
    ans += xgb.predict_proba(TEST)  # 각 클래스에 대한 예측확률

# print(list(c[np.where(select / 5 > 0)]))
score = list(score)
print(score)
print(sum(score) / 5)
ans /= 5
y_pred = np.argmax(ans, 1)
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)
