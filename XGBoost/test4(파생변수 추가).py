import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, EditedNearestNeighbours, TomekLinks
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
cols = ['u', 'g', 'r', 'i', 'z', 'redshift', 'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 'nObserve', 'nDetect', 'airmass_u', 'airmass_g', 'airmass_r', 'airmass_i', 'airmass_z']
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
y = train['class'] # 결과 레이블(class)
pd.set_option('display.max_rows', 500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)
TEST = test
df = pd.concat([train, TEST], axis=0)
df.fillna(-1, inplace=True)
df['d_dered_u'] = df['dered_u'] - df['u']
df['d_dered_g'] = df['dered_g'] - df['g']
df['d_dered_r'] = df['dered_r'] - df['r']
df['d_dered_i'] = df['dered_i'] - df['i']
df['d_dered_z'] = df['dered_z'] - df['z']
df['d_dered_rg'] = df['dered_r'] - df['dered_g']
df['d_dered_ig'] = df['dered_i'] - df['dered_g']
df['d_dered_zg'] = df['dered_z'] - df['dered_g']
df['d_dered_ri'] = df['dered_r'] - df['dered_i']
df['d_dered_rz'] = df['dered_r'] - df['dered_z']
df['d_dered_iz'] = df['dered_i'] - df['dered_z']
df['d_obs_det'] = df['nObserve'] - df['nDetect']
# print(df.head(100))
df = df.drop(['class', 'airmass_z', 'airmass_i', 'airmass_r', 'airmass_g', 'u', 'g', 'r', 'i', 'nDetect', 'd_dered_rg', 'd_dered_ri'], axis=1)
x = df[:320000]
print(x.shape)
TEST = df[320000:]
scaler = StandardScaler()
x = scaler.fit_transform(x)
TEST = scaler.transform(TEST)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
# ST = SMOTE(random_state=42, n_jobs=-1, k_neighbors=1, sampling_strategy="minority")
# train_x, train_y = ST.fit_resample(train_x, train_y)
evals = [(test_x, test_y)]
xgb = XGBClassifier(n_estimators=1000, n_jobs=-1, learning_rate=0.05, max_depth=50, subsample=0.65, objective="multi:softmax", random_state=42)
xgb.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
print("train acc : {}".format(xgb.score(train_x, train_y)))
print("test acc : {}".format(xgb.score(test_x, test_y)))
print(classification_report(test_y, xgb.predict(test_x), target_names=['class 0', 'class 1', 'class 2']))
# y_pred = np.argmax(xgb.predict_proba(TEST), axis=1) # 각 클래스에 대한 예측확률
# submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
# submission.to_csv('submission.csv', index=True)