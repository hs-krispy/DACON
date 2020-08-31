import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
y = train['class'] # 결과 레이블(class)
TEST = test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
# 데이터에서 20%를 test 데이터로 분리
evals = [(test_x, test_y)]
lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=12, num_leaves=4000, random_state=42, boosting_type="goss")
lgbm.fit(train_x, train_y, early_stopping_rounds=20, eval_set=evals)
print("acc: {}".format(lgbm.score(train_x, train_y))) # 훈련 데이터에 대한 정확도
print("acc: {}".format(lgbm.score(test_x, test_y))) # 테스트 데이터에 대한 정확도
y_pred = np.argmax(lgbm.predict_proba(TEST), axis=1) # 각 클래스에 대한 예측확률
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission5.csv', index=True)