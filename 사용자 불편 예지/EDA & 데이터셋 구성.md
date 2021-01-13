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

```python
missing_fill_val = {'quality_0': 0, 'quality_2': 0, 'quality_5': 0}
train_quality.fillna(missing_fill_val, inplace=True)
```

**train_problem_data**

- 사용자의 불만이 접수된 시간
- 불만이 접수된 시간 이후에도 train_err_data를 보면 에러 로그는 계속 발생했음을 알 수 있음

**test_err_data**

```python
print(test_err.isnull().sum())
# user_id     0
# time        0
# model_nm    0
# fwver       0
# errtype     0
# errcode	  4
test_err.dropna(inplace=True)
```

train_err_data와 마찬가지로 결측값이 있는 행을 삭제

**test_quality_data**

```python
print(test_quality.isnull().sum())
```

<img src="https://user-images.githubusercontent.com/58063806/104280404-33ecb580-54ef-11eb-91e8-d8965ffc2ada.png" width=20% />

train_quality_data와 마찬가지로 fwver 피처에 대해서는 삭제를 하고 나머지 피처들은  value들의 빈도를 구하고 가장 많은 빈도로 나타난 value로 대체 

```python
missing_fill_val = {'quality_0': '0', 'quality_1': '0', 'quality_2': '0', 'quality_5': '0'}
test_quality.fillna(missing_fill_val, inplace=True)
```

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

**train_err_data, test_err_data**

```python
encoder = LabelEncoder()


def encoding(train_df, test_df):
    train_list = train_df.unique().tolist()
    test_list = test_df.unique().tolist()
    union_list = list(set().union(train_list, test_list))
    encoder.fit(union_list)
    train_df = encoder.transform(train_df)
    test_df = encoder.transform(test_df)

    return train_df, test_df

# 확인 결과 train_err, test_err의 model_nm은 0 ~ 8로 모두 일치
train_err['model_nm'] = encoder.fit_transform(train_err['model_nm'])
test_err['model_nm'] = encoder.transform(test_err['model_nm'])

train_fwver = train_err['fwver'].unique().tolist()
test_fwver = test_err['fwver'].unique().tolist()
union_fwver = list(set().union(train_fwver, test_fwver))
print(len(train_fwver), len(test_fwver), len(union_fwver))
# 37 40 46
# 확인 결과 train_fwver과 test_fwver의 label 개수가 다름
train_err['fwver'], test_err['fwver'] = encoding(train_err['fwver'], test_err['fwver'])

train_errcode = train_err['errcode'].unique().tolist()
test_errcode = test_err['errcode'].unique().tolist()
union_errcode = list(set().union(train_errcode, test_errcode))
print(len(train_errcode), len(test_errcode), len(union_errcode))
# 2805 2955 4353
# 확인 결과 train_errcode, test_errcode의 label 개수가 다름
train_err['errcode'], test_err['errcode'] = encoding(train_err['errcode'], test_err['errcode'])
```

LabelEncoder를 통해 문자열로 구성되어 있는 피처들을 숫자형 카테고리 값으로 변환

<img src="https://user-images.githubusercontent.com/58063806/104254971-57960880-54bb-11eb-8c89-4cc9bc7b53a0.png" width=80% />

각각 9, 46, 4353개의 값을 가짐

```python
# ueser_id가 10000부터 24999까지 총 15000개가 연속적으로 존재.
train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000

sub_data = train_err[['user_id', 'errtype', 'errcode', 'model_nm']].values
dataset = np.zeros((train_user_number, 52))

count_errcode = np.zeros(4353)
pre_idx = 10000
for person_idx, errtype, errcode, model_nm in tqdm(sub_data):
    if pre_idx != person_idx:
        errccode = count_errcode.argmax()
        dataset[pre_idx - train_user_id_min][42] = errcode
        count_errcode = np.zeros(4353)
        pre_idx = person_idx
    # 에러타입 발생빈도
    dataset[person_idx - train_user_id_min][errtype - 1] += 1 
    # 해당 모델 사용 빈도
    dataset[person_idx - train_user_id_min][43 + model_nm] += 1 
    # 가장 많이 관측된 에러 코드 판별
    count_errcode[errcode] += 1 

dataset = pd.DataFrame(dataset)
dataset.to_csv("train_dataset.csv", index=False)

# test 데이터는 ueser_id가 30000부터 44998까지 총 14999개가 존재.
test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

sub_data = test_err[['user_id', 'errtype', 'errcode', 'model_nm']].values
dataset = np.zeros((test_user_number, 52))

count_errcode = np.zeros(4353)
pre_idx = 30000
for person_idx, errtype, errcode, model_nm in tqdm(sub_data):
    if pre_idx != person_idx:
        errccode = count_errcode.argmax()
        dataset[pre_idx - test_user_id_min][42] = errcode
        count_errcode = np.zeros(4353)
        pre_idx = person_idx
    # 에러타입 발생빈도
    dataset[person_idx - test_user_id_min][errtype - 1] += 1 
    # 해당 모델 사용 빈도
    dataset[person_idx - test_user_id_min][43 + model_nm] += 1 
    # 가장 많이 관측된 에러 코드 판별
    count_errcode[errcode] += 1 

dataset = pd.DataFrame(dataset)
dataset.to_csv("test_dataset.csv", index=False)
```

기존의 train_err 데이터에서 user_id, errtype, errcode, model_nm을 이용해서 새로운 데이터셋 생성

동일한 조건에서 해당 데이터셋으로 성능 평가

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LGBMClassifier(random_state=42)
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
# validation score - 0.8145382
# after standard scaling validation score - 0.813758
# submission score - 0.81732
```

AUC가 0.01 정도 상승한 결과를 보임

**train_quality_data**

```python
for i in range(13):
    print("quality{} unique value count:".format(i), len(train_quality['quality_{}'.format(i)].unique()))
```

<img src="https://user-images.githubusercontent.com/58063806/104274902-34804e80-54e5-11eb-9fbc-e7ea4633a67b.png" width=30%/>

```python
# ,가 들어가 있어 산술연산이 불가능했던 column들에 대해 ,를 제거하고 다시 int형으로 변환
for i in (5, 7, 8, 9, 10):
    train_quality['quality_{}'.format(i)].replace(',', '', regex=True, inplace=True)
    train_quality['quality_{}'.format(i)] = train_quality['quality_{}'.format(i)].astype(int)

# 각 user_id의 시스템 퀄리티 별로 가장 많이 발생했던 값으로 설정
dataset = np.zeros((train_user_number, 13))
unique_id = train_quality['user_id'].unique().tolist()
for id in unique_id:
    for idx in range(13):
        dataset[id - train_user_id_min, idx] = train_quality.loc[train_quality['user_id'] == id]['quality_{}'.format(idx)].value_counts().index[0]

dataset = pd.DataFrame(dataset)
dataset.to_csv("train_quality.csv")
```

<img src="https://user-images.githubusercontent.com/58063806/104280022-9b563580-54ee-11eb-8e40-9a7daaacc396.png" width=70% />

**test_quality_data**

train_quality_data와 동일한 방식으로 진행

```python
for i in range(13):
    test_quality['quality_{}'.format(i)].replace(',', '', regex=True, inplace=True)
    test_quality['quality_{}'.format(i)] = test_quality['quality_{}'.format(i)].astype(int)

dataset = np.zeros((test_user_number, 13))
unique_id = test_quality['user_id'].unique().tolist()
for id in unique_id:
    for idx in range(13):
        dataset[id - test_user_id_min, idx] = test_quality.loc[test_quality['user_id'] == id]['quality_{}'.format(idx)].value_counts().index[0]

dataset = pd.DataFrame(dataset)
dataset.to_csv("test_quality.csv")
```

<img src="https://user-images.githubusercontent.com/58063806/104281467-e83b0b80-54f0-11eb-9cdc-33364e373c48.png" width=70%/>

```python
train = pd.read_csv(PATH+'train_dataset.csv')
test = pd.read_csv(PATH+'test_dataset.csv')
train_quality = pd.read_csv(PATH+'train_quality.csv')
test_quality = pd.read_csv(PATH+'test_quality.csv')
train = pd.concat([train, train_quality], axis=1, ignore_index=True)
train.drop(train.columns[52], axis='columns', inplace=True)
test = pd.concat([test, test_quality], axis=1, ignore_index=True)
test.drop(test.columns[52], axis='columns', inplace=True)
```

기존에 생성했던 train_dataset과 train_quality, test_dataset과 test_quality로 새로운 dataset 생성

동일한 조건에서 해당 데이터셋으로 성능 평가

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LGBMClassifier(random_state=42)
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
# validation score - 0.8147328
# after standard scaling validation score - 0.8152645999999999
# after standard scaling submission score - 0.8137198286
# submission score - 0.8162371843	
```

train_quality, test_quality 데이터를 결합해서 생성한 새로운 데이터셋은 0.001 정도 AUC 감소를 보임

**train_err_data, test_err_data**

err_data에서 time 피처를 활용해서 errors_per_day라는 새로운 피처를 생성

```python
def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
    return dt.datetime(year, month, day)

# 각 유저별 하루동안 발생한 평균 에러의 갯수
def cal_errors_per_day(df):
    df['datetime'] = df['time'].apply(make_datetime)
    unique_date = df.groupby('user_id')['datetime'].unique().values
    # 각 유저별 err 발생 날짜의 갯수
    count_date = []
    for i in unique_date:
        count_date.append(len(i))

    # 각 유저별 발생한 에러의 횟수
    id_error = df.groupby('user_id')['errtype'].count().values
    avg_err = []
    for idx, val in enumerate(count_date):
        avg_err.append(id_error[idx] / val)

    return avg_err


train_avg_err = cal_errors_per_day(train_err)
test_avg_err = cal_errors_per_day(test_err)
```

errors_per_day 피처를 추가한 데이터셋을 같은 조건에서 학습을 진행

```python
lgb = LGBMClassifier(random_state=42)

def validation(model, x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = []
    for train_idx, valid_idx in skf.split(x, y):
        train_x, valid_x = x.iloc[train_idx], x.iloc[valid_idx]
        train_y, valid_y = y[train_idx], y[valid_idx]
        evals = [(valid_x, valid_y)]
        model.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
        valid_prob = model.predict_proba(valid_x)[:, 1]
        auc_score = roc_auc_score(valid_y, valid_prob)
        score.append(auc_score)

    return np.mean(score)

avg_AUC = validation(lgb, X, y)
print(avg_AUC)

# validation score - 0.814518
# submission score - 0.8138071294	
```



### feature_importance

```python
# 피처 중요도 그래프의 시각화를 용이하게 하기 위해 feature label 생성 
label = sorted(train_err['errtype'].unique().tolist())
for i, val in enumerate(label):
    label[i] = "errtype_" + str(val)
label.append('errcode')
model = sorted(train_err['model_nm'].unique().tolist())
for val in model:
    label.append("model_nm_" + str(val))
for i in range(13):
    label.append("quality_{}".format(i))
```

<img src="https://user-images.githubusercontent.com/58063806/104318808-42ed5b00-5523-11eb-9b78-9d0e8d836c52.png" width=100% />

```python
train.columns = label
train_prob = pd.read_csv(PATH+'train_problem_data.csv')
problem = np.zeros(15000)
problem[train_prob.user_id.unique()-10000] = 1
problem = pd.DataFrame(problem)
X = train
y = problem
lgb = LGBMClassifier(random_state=42)

def train(model, X, y):
    model.fit(X, y)

def feature_importance(model):
    fig, ax = plt.subplots(figsize=(15, 9))
    plot_importance(model, ignore_zero=False, ax=ax)
    plt.savefig('feature_importance.png')

train(lgb, X, y)
feature_importance(lgb)
```

<img src="https://user-images.githubusercontent.com/58063806/104322586-7979a480-5528-11eb-9743-6b2b1de24f3e.png" width=100% />

피처 중요도 차트를 본 결과 quality 피처들 대부분이 낮은 중요도를 보임 (Feature selection 요망)

<img src="https://user-images.githubusercontent.com/58063806/104397794-dc068b00-5590-11eb-8d30-1ae927a24892.png" width=100% />quality를 제외하고 err data만 이용한 결과

<img src="https://user-images.githubusercontent.com/58063806/104414954-71ffdd00-55b4-11eb-9509-36c1e4f5d293.png" width=100% />

err_data에 errors_per_day 피처를 추가한 결과 (추가된 errors_per_day 피처가 높은 중요도를 보임)
