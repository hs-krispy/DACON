import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from xgboost import XGBClassifier
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
y = train['class'] # 결과 레이블(class)
test_x = test
class0 = len(y[y == 0])
class1 = len(y[y == 1])
class2 = len(y[y == 2])
total = len(y)
scaler = StandardScaler()
x = scaler.fit_transform(train_x)
TEST = scaler.transform(test_x)
BS = BorderlineSMOTE(random_state=42, n_jobs=-1, k_neighbors=3)
x, y = BS.fit_resample(x, y)
print(pd.value_counts(y))
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
evals = [(test_x, test_y)]
xgb = XGBClassifier(n_estimators=600, n_jobs=-1, learning_rate=0.05, subsample=0.65, max_depth=50, objective="multi:softmax", random_state=42)
xgb.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
# ans_pred += forest_pred
# ans_pred += xgb_pred
# ans_pred /= 2.0
print("acc: {}".format(xgb.score(train_x, train_y)))
print("acc: {}".format(xgb.score(test_x, test_y)))
y_pred = np.argmax(xgb.predict_proba(TEST), axis=1)
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)
# 테스트 데이터에 대해 94.6% 정도의 accuracy가 나왔지만 실제로는 92.215%가 나옴