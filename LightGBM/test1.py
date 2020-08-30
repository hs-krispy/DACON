import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
train_y = train['class'] # 결과 레이블(class)
TEST = test
# xgb_param = { 'n_estimators': [100, 200, 300],
#                'max_depth': [7, 10, 13, 15],
#                'learning_rate': [0.01, 0.05, 0.1, 0.15]}

lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.15, max_depth=17, num_leaves=131072).fit(train_x, train_y)
# XGBoost와 비슷한 값으로 진행(max_depth는 최대치로 진행)
print("acc: {}".format(lgbm.score(train_x, train_y)))
y_pred = np.argmax(lgbm.predict_proba(TEST), axis=1) # 각 클래스에 대한 예측확률
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission5.csv', index=True)