import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
y = train['class'] # 결과 레이블(class)
test_x = test
scaler = StandardScaler()
x = scaler.fit_transform(train_x)
TEST = scaler.transform(test_x)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
xgb = XGBClassifier(n_estimators=400, n_jobs=-1, learning_rate=0.05, subsample=0.65, max_depth=50, objective="multi:softmax", random_state=42)
evals = [(test_x, test_y)]
xgb.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
print("acc: {}".format(xgb.score(train_x, train_y)))
print("acc: {}".format(xgb.score(test_x, test_y)))
y_pred = np.argmax(xgb.predict_proba(TEST), axis=1) # 각 클래스에 대한 예측확률
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)