## 딥러닝(midterm)

2020-2 딥러닝 수업의 경진대회

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
Train = pd.read_csv("dataset/trainset.csv")
Test = pd.read_csv("dataset/testset.csv")
print(Train.isnull().sum()) # 결측값 확인
print(Train.value_counts("HI")) # 클래스별 데이터 갯수 확인
```

**클래스별 데이터 갯수**

<img src="https://user-images.githubusercontent.com/58063806/95991015-36476100-0e67-11eb-912e-ad26fd6215d6.JPG" width=15% />

```python
scaler = StandardScaler()
x = scaler.fit_transform(x)
logi = LogisticRegression()
xgb = XGBClassifier()
forest = RandomForestClassifier()
svm = svm.SVC()
m = [logi, forest, svm, xgb]
kf = KFold(n_splits=10, random_state=123, shuffle=True)
res = []
for model in m:
    i = 0
    acc = np.zeros(10)
    for train_idx, test_idx in kf.split(x):
        train_x, test_x = x[train_idx], x[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]
        model.fit(train_x, train_y)
        accuracy = model.score(test_x, test_y)
        acc[i] = accuracy
        i += 1
    res.append([model, acc.mean(), acc.std()])
print(res)
```

데이터에 대해 StandardScaling을 진행한 후 LogisticRegression, XGBoost, RandomForest, SVM 4가지 모델에 대해 kfold를 진행후 평균 acc 값과 표준 편차 값을 구함

<img src="https://user-images.githubusercontent.com/58063806/95990556-a3a6c200-0e66-11eb-957a-f2801deb0c3e.JPG" width=100% />

표준편차는 4가지의 모델에서 큰 차이가 없으며 xgboost를 사용했을때 가장 높은 acc가 나타남 

```python
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
```

<img src="https://user-images.githubusercontent.com/58063806/95991010-35aeca80-0e67-11eb-8a5f-9734691a4582.JPG" width=100% />

클래스의 비율을 고려한 StratifiedKFold를 사용함, 기존의 KFold와 큰 차이는 없음

**xgboost를 사용해서 진행**

