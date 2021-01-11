import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
train_y = train['class'] # 결과 레이블(class)
test_x = test
# xgb_param = { 'n_estimators': [100, 200, 300],
#                'max_depth': [7, 10, 13, 15],
#                'learning_rate': [0.01, 0.05, 0.1, 0.15]}
xgb = XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=72).fit(train_x, train_y)
print("acc: {}".format(xgb.score(train_x, train_y)))
y_pred = np.argmax(xgb.predict_proba(test_x), axis=1) # 각 클래스에 대한 예측확률
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)