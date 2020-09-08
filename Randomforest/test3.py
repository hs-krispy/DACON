import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns=['class'], axis=1) # class 열을 삭제한 새로운 객체
train_y = train['class'] # 결과 레이블(class)
TEST = test
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ans = []
score = []
for train_idx, test_idx in skf.split(train_x, train_y):
    x_train, x_test = train_x.iloc[train_idx], train_x.iloc[test_idx]
    y_train, y_test = train_y.iloc[train_idx], train_y.iloc[test_idx]
    evals = [(x_test, y_test)]
    forest = RandomForestClassifier(criterion="entropy", n_jobs=-1, max_depth=40, n_estimators=250, max_features=11, random_state=42, verbose=True).fit(x_train, y_train)
    print("acc: {}".format(forest.score(x_test, y_test)))
    score.append(forest.score(x_test, y_test))
    print(confusion_matrix(y_test, forest.predict(x_test)))
    y_pred = np.argmax(forest.predict_proba(TEST), axis=1) # 각 클래스에 대한 예측확률
    ans.append(y_pred)

score = list(score)
print(score)
print(sum(score) / 5)
idx = score.index(max(score))
submission = pd.DataFrame(data=ans[idx], columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)