## Randomforest

decision tree(결정 트리)를 기반으로 만들어진 알고리즘으로 여러 개의 결정 트리 classifier가 생성되고 각자의 방식으로 데이터를 sampling하여 개별적으로 학습하고 최종적으로  voting(투표를 통한 결과 도출)을 통해 예측을 수행하는 알고리즘

<img src="https://user-images.githubusercontent.com/58063806/91415654-5310e080-e889-11ea-808d-5378a5f2f12c.JPG" width=70% />

(이미지 출처 - https://lsjsj92.tistory.com/542)

**부트스트래핑(bootstraping) - 각각의 classifier가 학습할 dataset을 original data에서 sampling해서 가져온 것(데이터가 중복될 수 있음)**

### 장점

- Overfitting이 잘 되지 않음
- Training이 빠름

### 단점

- Memory 사용량이 굉장히 많음(Decision Tree를 만드는 데 많이 메모리를 사용하기 때문)
- Training data의 양이 증가해도 급격한 성능의 향상을 기대하기 어려움

### 시도한 방식

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
train = pd.read_csv("/content/drive/My Drive/DACON/train.csv", index_col=0)
test = pd.read_csv("/content/drive/My Drive/DACON/test.csv", index_col=0)
sample_submission = pd.read_csv("/content/drive/My Drive/DACON/sample_submission.csv", index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
train_y = train['class'] # 결과 레이블(class)
test_x = test
n_estimators = [int (x) for x in np.linspace(100, 1000, 11)]
max_depth = [int (x) for x in np.linspace(10, 100, 11)]
min_samples_split = [2, 4, 8]
min_samples_leaf = [1, 2, 4]
# 파라미터 후보군
random_grid = { 'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf }
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=1, cv=3, verbose=2, random_state=42, n_jobs=-1) # 랜덤하게 실행시켜보고 성능을 평가
rf_random.fit(train_x, train_y)
rf_random.best_params_ # 최적의 파라미터를 출력
```

최적의 파라미터를 찾고자 RandomForestRegressor, RandomizedSearchCV를 이용 

n_estimators - 포레스트의 트리 수
max_depth - 각 의사 결정 트리의 최대 레벨 수
min_samples_split - 노드가 분할되기 전에 노드에 배치된 최소 데이터 포인트 수
min_samples_leaf - 리프 노드에서 허용되는 최소 데이터 포인트 수