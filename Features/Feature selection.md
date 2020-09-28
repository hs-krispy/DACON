## Feature selection

많은 특징 데이터에서 학습에 중요한 영향을 주는 특징데이터만 선택하기 위함

### 분산에 의한 선택

예측 모형에서 중요한 특징데이터는 종속데이터와의 상관관계가 크고 예측에 도움되는 데이터인데 특징데이터의 값 자체가 표본에 따라 그다지 변하지 않으면 종속데이터 예측에도 도움이 되지 않을 가능성이 높음

**(표본 변화에 따른 데이터 값의 변화 즉, 분산이 기준치보다 낮은 특징 데이터는 사용하지 않는 방법)**

**예시**

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(1e-5)
X_train_sel = selector.fit_transform(X_train)
X_test_sel = selector.transform(X_test)
X_train_sel.shape
```

### 단일 변수 선택

각각의 독립변수를 하나만 사용한 예측모형의 성능을 이용하여 분류성능이나 상관관계가 가장 높은 변수만 선택하는 방법

- chi2 : 카이제곱 검정 통계값
- f_classif : 분산분석(ANOVA) F검정 통계값
- mutual_info_classif : 상호정보량(mutual information)

**예시**

```python
from sklearn.feature_selection import chi2, SelectKBest
selector = SelectKBest(chi2, k=14330)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
```

### 모델을 이용한 특성 중요도 계산

특성 중요도(feature importance)를 계산할 수 있는 랜덤포레스트 등의 모델을 이용해서 피처를 정제(최종 분류는 다른 모델을 사용할 수도 있음)

**예시**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
model_sel = ExtraTreesClassifier(n_estimators=50).fit(X_train, y_train)
selector = SelectFromModel(model_sel, prefit=True, max_features=14330)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)
```

### 재귀적 특성 제거(Recursive Feature Elimination)

- 특성을 하나도 선택하지 않은 상태로 시작해서 종료 조건에 도달할 때까지 특성을 하나씩 추가하는 방법

- 모든 특성을 가지고 시작해서 종료 조건에 도달할 때까지 특성을 하나씩 제거해가는 방법

**위의 다른 방법들보다 계산 비용이 더 많이 들지만 보다 정확한 결과를 위해 RFE를 적용해보았음**

**예시**

```python
xgb = XGBClassifier(n_estimators=1000, n_jobs=-1, learning_rate=0.05, subsample=0.65, max_depth=50, objective="multi:softmax", random_state=42)
select = RFE(xgb, n_features_to_select=40, verbose=True).fit(x, y)
# 모든 특성을 가지고 시작해서 제거해나감
x_select = select.transform(x)
TEST_select = select.transform(TEST)
```

### permutation_importance

피처들이 결과에 어떠한 영향을 미치는지 판별

```python
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
evals = [(test_x, test_y)]
xgb = XGBClassifier(n_estimators=1000, n_jobs=-1, learning_rate=0.05, subsample=0.65, max_depth=50, objective="multi:softmax", random_state=42).fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
results = permutation_importance(xgb, test_x, test_y, n_jobs=-1, n_repeats=1, scoring='accuracy')
importance = results.importances # 반복 횟수에 따라 importances_mean, importance_std 도 가능
for i, v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i, v))
```

