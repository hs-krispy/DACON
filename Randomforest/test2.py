import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
train_y = train['class'] # 결과 레이블(class)
test_x = test
# n_estimators = [int (x) for x in np.linspace(200, 400, 11)]
# max_depth = [int (x) for x in np.linspace(70, 90, 11)]
# min_samples_split = [3, 4, 5]
# min_samples_leaf = [2, 3]
# 파라미터 후보군
# random_grid = { 'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf }
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# rf_random.best_params_
forest = RandomForestClassifier(n_estimators=260, max_depth=72, min_samples_leaf=3, min_samples_split=3).fit(train_x, train_y)
print("acc: {}".format(forest.score(train_x, train_y)))
y_pred = np.argmax(forest.predict_proba(test_x), axis=1) # 각 클래스에 대한 예측확률
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)