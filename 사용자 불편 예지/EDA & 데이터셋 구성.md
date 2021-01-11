## 사용자 불편 예지

### EDA

##### 결측값 처리

**train_err_data.csv**

- 시스템에 발생한 에러 로그

```python
train_err = pd.read_csv(PATH+'train_err_data.csv')
print(train_err.isnull().sum())
# user_id     0
# time        0
# model_nm    0
# fwver       0
# errtype     0
# errcode     1

# errcode에 하나의 결측값 확인
print(train_err.value_counts('errcode'))
```

<img src="https://user-images.githubusercontent.com/58063806/104117060-6b2d5c00-5361-11eb-85de-ac6c980dd509.png" width=25% />

상당히 많은 양의 errcode가 있는 것을 볼 수 있음

그 결과 하나의 errcode로 결측값을 대체하기 보다는 삭제하는 방향을 선택

```python
train_err.dropna(inplace=True)
print(train_err.isnull().sum())
# user_id     0
# time        0
# model_nm    0
# fwver       0
# errtype     0
# errcode     0
```

**train_quality_data**

- 시스템 퀄리티 로그
- 12행 단위로 시간이 변함

```python
train_quality = pd.read_csv(PATH+'train_quality_data.csv')
print(train_quality.isnull().sum())
```

<img src="https://user-images.githubusercontent.com/58063806/104149828-151cef00-541b-11eb-9dc7-949d51ef1c41.png" width=20% />

여러 피처에서 상당히 많은 양의 결측값이 관측

```python
# fwver 피처는 예상이 불가하므로 삭제
train_quality.dropna(subset=['fwver'], inplace=True) 
```

<img src="https://user-images.githubusercontent.com/58063806/104150486-4b5b6e00-541d-11eb-8504-33ca3f369057.png" width=20%/>

남은 피처들에 대해서는 value들의 빈도를 구하고 가장 많은 빈도로 나타난 value로 대체 

**train_problem_data**

- 사용자의 불만이 접수된 시간
- 불만이 접수된 시간 이후에도 train_err_data를 보면 에러 로그는 계속 발생했음을 알 수 있음



### 기본적인 데이터셋 생성

```python
train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000

# user_id와 errtype만을 사용하여 데이터 셋 생성
id_error = train_err[['user_id', 'errtype']].values
error = np.zeros((train_user_number, 42))

# 각 유저에 대한 해당 에러의 발생빈도
for person_idx, err in tqdm(id_error):
    error[person_idx - train_user_id_min, err - 1] += 1
    
problem = np.zeros(15000)
# person_idx의 problem이 한 번이라도 발생했다면 1, 없다면 0
problem[train_prob.user_id.unique()-10000] = 1

x = error
y = problem

# train data와 마찬가지로 test data 생성
test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

id_error = test_err[['user_id', 'errtype']].values
test_x = np.zeros((test_user_number, 42))
for person_idx, err in tqdm(id_error):
    test_x[person_idx - test_user_id_min, err - 1] += 1
```

### 테스팅

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(random_state=42)
score = []
for train_idx, valid_idx in skf.split(x, y):
    train_x, valid_x = x[train_idx], x[valid_idx]
    train_y, valid_y = y[train_idx], y[valid_idx]
    evals = [(valid_x, valid_y)]
    model.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
    valid_prob = model.predict_proba(valid_x)[:, 1]
    auc_score = roc_auc_score(valid_y, valid_prob)
    score.append(auc_score)

print(np.mean(score))
# validation score - 0.8032662000000002
# submission score - 0.80228

model = Randomforest(random_state=42)
# validation score - 0.79861015
model = ExtraTreesClassifier(random_state=42)
# validation score - 0.8000375500000001
model = LGBMClassifier(random_state=42)
# validation score - 0.8074797
# after standard scaling validation score - 0.8080327999999998
# submission score - 0.80732
```



### 추가적인 데이터 

**train_quality_data**

```python
# ,가 들어가 있어 산술연산이 불가능했던 column들에 대해 ,를 제거하고 다시 int형으로 변환
for i in (5, 7, 8, 9, 10):
    train_quality['quality_{}'.format(i)].replace(',', '', regex=True, inplace=True)
    train_quality['quality_{}'.format(i)] = train_quality['quality_{}'.format(i)].astype(int)

# 각 user_id에 대한 시스템 퀄리티 로그의 합계
print(train_quality[train_quality.columns.difference(['time', 'fwver'])].groupby('user_id').sum())
quality = np.array(train_quality.groupby('user_id').sum()[1:].values)
```

<img src="https://user-images.githubusercontent.com/58063806/104189331-df4d2a00-545d-11eb-80ad-8df16fe8a30a.png" width=100% />

