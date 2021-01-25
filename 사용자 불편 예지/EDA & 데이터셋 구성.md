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
train_quality['quality_0'].fillna("N", inplace=True)
train_quality['quality_2'].fillna("N", inplace=True)
train_quality['quality_5'].fillna("N", inplace=True)
```

각 피처들이 범주형 변수라고 판단하고 결측값을 N으로 채움

**train_problem_data**

- 사용자의 불만이 접수된 시간
- 불만이 접수된 시간 이후에도 train_err_data를 보면 에러 로그는 계속 발생했음을 알 수 있음

**test_err_data**

test_err에는 user_id 43262에 대한 정보가 없음(에러 발생 X)

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

```python
test_quality['quality_0'].fillna("N", inplace=True)
test_quality['quality_1'].fillna("N", inplace=True)
test_quality['quality_2'].fillna("N", inplace=True)
test_quality['quality_5'].fillna("N", inplace=True)
```

train_quality와 마찬가지로 결측값을 모두 N으로 채움기본적인 데이터셋 생성

```python

def fill_null(df_qual, df_err, which):
    null_id = df_qual[df_qual['fwver'].isnull() == True]['user_id'].unique()
    for id in null_id:
        if id == 43262:
            continue
        print(df_err[df_err['user_id'] == id]['fwver'].value_counts())
        val = df_err[df_err['user_id'] == id]['fwver'].unique()
        if len(val) > 1:
            start_time = df_qual[df_qual['user_id'] == id]['time'].unique()[0]
            err_time = df_err[df_err['user_id'] == id][['time', 'fwver']].values
            for et in err_time:
                if et[0] > start_time:
                    df_qual.loc[df_qual['user_id'] == id, 'fwver'] = et[1]
        df_qual.loc[df_qual['user_id'] == id, 'fwver'] = val[0]
    if which == "test":
        index = df_qual[df_qual['user_id'] == 43262].index
        df_qual.drop(index=index, inplace=True)
    print(df_qual.isnull().sum())
    df_qual.to_csv("filled_{}_quality.csv".format(which), index=False)


fill_null(train_quality, train_err, "train")
fill_null(test_quality, test_err, "test")
```

err data를 참고해서 user_id를 통해 quality data의 fwver 결측값을 채워넣음 

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


def encoding(train_err, test_err, train_qual=None, test_qual=None):
    train_err_list = train_err.unique().tolist()
    test_err_list = test_err.unique().tolist()
    if train_qual is not None:
        train_qual_list = train_qual.unique().tolist()
        test_qual_list = test_qual.unique().tolist()
        union_list = list(set().union(train_err_list, train_qual_list, test_err_list, test_qual_list))
    else:
        union_list = list(set().union(train_err_list, test_err_list))
    print(len(union_list))
    encoder.fit(union_list)
    encode_train_err = encoder.transform(train_err)
    encode_test_err = encoder.transform(test_err)
    if train_qual is not None:
        encode_train_qual = encoder.transform(train_qual)
        encode_test_qual = encoder.transform(test_qual)
        return encode_train_err, encode_test_err, encode_train_qual, encode_test_qual
    return encode_train_err, encode_test_err


train_err['model_nm'], test_err['model_nm'] = encoding(train_err['model_nm'], test_err['model_nm'])
train_err['fwver'], test_err['fwver'], train_quality['fwver'], test_quality['fwver'] = encoding(train_err['fwver'], test_err['fwver'], train_quality['fwver'], test_quality['fwver'])
train_err['errcode'], test_err['errcode'] = encoding(train_err['errcode'], test_err['errcode'])
```

LabelEncoder를 통해 문자열로 구성되어 있는 피처들을 숫자형 카테고리 값으로 변환

각각 9, 48, 4353개의 값을 가짐

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



**각 유저별 하루동안 발생한 평균 에러의 갯수**

```python
def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
	
    return dt.datetime(year, month, day)

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


def cal_errors_per_day2(df):
    df['datetime'] = df['time'].apply(make_datetime)
    unique_date = df.groupby('user_id')['datetime'].unique().values
    # 각 유저별 err 발생 날짜의 갯수
    count_date = []
    for i in unique_date:
        count_date.append(len(i))

    # 각 유저별 발생한 에러의 횟수
    id_error = df.groupby('user_id')['errtype'].count().values
    avg_err = []
    for idx, val in enumerate(count_date[:13262]):
        avg_err.append(id_error[idx] / val)
    # user_id 43262에 대한 예외 처리 (에러가 발생하지 않았음으로 0으로 처리)
    avg_err.append(0)
    for idx, val in enumerate(count_date[13262:]):
        avg_err.append(id_error[13262 + idx] / val)

    return avg_err


train_avg_err = cal_errors_per_day(train_err)
test_avg_err = cal_errors_per_day2(test_err)
```

**각 유저별 에러가 발생한 간격**

```python
def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
    hour = int(x[8:10])
    mim = int(x[10:12])
    sec = int(x[12:14])
    return dt.datetime(year, month, day, hour, mim, sec)

def cal_time_interval(df):
    df['datetime'] = df['time'].apply(make_datetime)
    # 각 유저별 에러발생 시간
    times = df['datetime'].values
    # 각 유저별 에러발생 횟수
    count_time = df.groupby('user_id')['datetime'].count().values
    time_interval = []
    init = 0
    for c in count_time:
        if c == 1:
            time_interval.append(0)
            init += 1
            continue
        sum = 0
        pre_t = 0
        for idx, t in enumerate(times[init: init + c]):
            if idx == 0:
                pre_t = t
                continue
            sum += t - pre_t
            pre_t = t
        # 각 유저별 평균적으로 에러가 발생하는 간격
        time_interval.append(sum / (c - 1))
        init += c

    return time_interval


def cal_time_interval2(df):
    df['datetime'] = df['time'].apply(make_datetime)
    # 각 유저별 에러발생 시간
    times = df['datetime'].values
    # 각 유저별 에러발생 횟수
    count_time = df.groupby('user_id')['datetime'].count().values
    time_interval = []
    init = 0
    for c in count_time:
        if c == 1:
            time_interval.append(0)
            init += 1
            continue
        sum = 0
        pre_t = 0
        for idx, t in enumerate(times[init: init + c]):
            if idx == 0:
                pre_t = t
                continue
            sum += t - pre_t
            pre_t = t
        # 각 유저별 평균적으로 에러가 발생하는 간격
        time_interval.append(sum / (c - 1))
        # user_id 43262에 대한 예외 처리 (에러가 발생하지 않았음으로 0으로 처리)
        if len(time_interval) == 13262:
            time_interval.append(0)
        init += c

    return time_interval


train_time_interval = cal_time_interval(train_err)
test_time_interval = cal_time_interval2(test_err)
```

**각 유저별로 errtype들의 발생빈도, 가장 많이 발생한 errcode, 각 model들의 사용빈도**

```python
sub_data = train_err[['user_id', 'errtype', 'errcode', 'model_nm']].values
dataset = np.zeros((train_user_number, 53))

count_errcode = np.zeros(4353)
pre_idx = 10000
for idx, val in enumerate(train_avg_err):
    dataset[idx][51] = val
for idx, val in enumerate(train_time_interval):
    dataset[idx][52] = val
for person_idx, errtype, errcode, model_nm in tqdm(sub_data):
    if pre_idx != person_idx:
        errccode = count_errcode.argmax()
        dataset[pre_idx - train_user_id_min][41] = errcode
        count_errcode = np.zeros(4353)
        pre_idx = person_idx
    if errtype > 29:
        dataset[person_idx - train_user_id_min][errtype - 2] += 1
    else:
        # 각 에러 타입 발생빈도
        dataset[person_idx - train_user_id_min][errtype - 1] += 1
	# 각 모델의 사용 빈도
    dataset[person_idx - train_user_id_min][42 + model_nm] += 1
    # 가장 많이 관측된 에러 코드 판별
    count_errcode[errcode] += 1  

dataset = pd.DataFrame(dataset, columns=label)
dataset.to_csv("train_dataset.csv", index=False)

sub_data = test_err[['user_id', 'errtype', 'errcode', 'model_nm']].values
dataset = np.zeros((test_user_number, 53))

count_errcode = np.zeros(4353)
pre_idx = 30000
for idx, val in enumerate(test_avg_err):
    dataset[idx][51] = val
for idx, val in enumerate(test_time_interval):
    dataset[idx][52] = val
for person_idx, errtype, errcode, model_nm in tqdm(sub_data):
    if pre_idx != person_idx:
        errccode = count_errcode.argmax()
        dataset[pre_idx - test_user_id_min][41] = errcode
        count_errcode = np.zeros(4353)
        pre_idx = person_idx
    if errtype > 29:
        dataset[person_idx - test_user_id_min][errtype - 2] += 1
    else:
        # 에러타입 발생빈도
        dataset[person_idx - test_user_id_min][errtype - 1] += 1
    # 해당 모델 사용 빈도    
    dataset[person_idx - test_user_id_min][42 + model_nm] += 1  
    # 가장 많이 관측된 에러 코드 판별
    count_errcode[errcode] += 1  

dataset = pd.DataFrame(dataset, columns=label)
dataset.to_csv("test_dataset.csv", index=False)
```

해당 데이터셋으로 동일조건에서 학습

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
# validation score - 0.8129683
# after standard scale score - 0.81278475
# submission score - 0.8136799738	
```

```python
print(train_err.groupby('user_id')['model_nm'].nunique())
```

데이터를 더 살펴본 결과 대부분의 유저에서 하나의 모델만 사용하는 것으로 나타남

error_time_interval의 수치들을 0 ~ 1사이의 값으로 scaling해서 다시 데이터셋 생성

```python
label = sorted(train_err['errtype'].unique().tolist())
for i, val in enumerate(label):
    label[i] = "errtype_" + str(val)
label.extend(['errcode', 'model_nm', 'errors_per_day', 'error_time_interval'])

# 각 유저별 하루동안 발생한 평균 에러의 갯수
def cal_errors_per_day(df, which):
    df['datetime'] = df['time'].apply(make_datetime)
    unique_date = df.groupby('user_id')['datetime'].unique().values
    # 각 유저별 err 발생 날짜의 갯수
    count_date = []
    for i in unique_date:
        count_date.append(len(i))

    # 각 유저별 발생한 에러의 횟수
    id_error = df.groupby('user_id')['errtype'].count().values
    avg_err = []
    if which == "train":
        for idx, val in enumerate(count_date):
            avg_err.append(id_error[idx] / val)
    else:
        for idx, val in enumerate(count_date[:13262]):
            avg_err.append(id_error[idx] / val)
        # user_id 43262에 대한 예외 처리 (에러가 발생하지 않았음으로 0으로 처리)
        avg_err.append(0)
        for idx, val in enumerate(count_date[13262:]):
            avg_err.append(id_error[13262 + idx] / val)

    return avg_err


train_avg_err = cal_errors_per_day(train_err, "train")
test_avg_err = cal_errors_per_day(test_err, "test")


# 각 유저별 에러가 발생한 간격
def cal_time_interval(df, which):
    df['datetime'] = df['time'].apply(make_datetime2)
    # 각 유저별 에러발생 시간
    times = df['datetime'].values
    # 각 유저별 에러발생 횟수
    count_time = df.groupby('user_id')['datetime'].count().values
    time_interval = []
    init = 0

    for c in count_time:
        if c == 1:
            time_interval.append(0)
            init += 1
            continue
        sum = 0
        pre_t = 0
        for idx, t in enumerate(times[init: init + c]):
            if idx == 0:
                pre_t = t
                continue
            sum += t - pre_t
            pre_t = t
        # 각 유저별 평균적으로 에러가 발생하는 간격
        time = sum / (c - 1)
        time = str(time)
        time = float(time[:-12])
        time_interval.append(time)
        if which == "test":
            # user_id 43262에 대한 예외 처리 (에러가 발생하지 않았음으로 0으로 처리)
            if len(time_interval) == 13262:
                time_interval.append(0)
        init += c
        
    # 값을 0 ~ 1 사이로 스케일링
    scaler = MinMaxScaler()
    time_interval = np.array(time_interval).reshape(-1, 1)
    time_interval = scaler.fit_transform(time_interval)
    return time_interval


train_time_interval = cal_time_interval(train_err, "train")
test_time_interval = cal_time_interval(test_err, "test")


def making_dataset(df, user_number, user_id_min, avg_err, time_interval):
    sub_data = df[['user_id', 'errtype', 'errcode', 'model_nm']].values
    dataset = np.zeros((user_number, 53))

    count_errcode = np.zeros(4353)
    pre_idx = user_id_min
    for idx, (avg_err, time_interval) in enumerate(zip(avg_err, time_interval)):
        dataset[idx][51] = avg_err
        dataset[idx][52] = time_interval
    for person_idx, errtype, errcode, model_nm in tqdm(sub_data):
        if pre_idx != person_idx:
            errcode = count_errcode.argmax()
            dataset[pre_idx - user_id_min][41] = errcode
            count_errcode = np.zeros(4353)
            pre_idx = person_idx
        if errtype > 29:
            dataset[person_idx - user_id_min][errtype - 2] += 1
        else:
            dataset[person_idx - user_id_min][errtype - 1] += 1  # 에러타입 발생빈도
        # 각 모델의 사용 빈도
        dataset[person_idx - user_id_min][42 + model_nm] += 1
        count_errcode[errcode] += 1  # 가장 많이 관측된 에러 코드 판별

    dataset = pd.DataFrame(dataset, columns=label)
    dataset.to_csv("train_dataset.csv", index=False)
    return dataset



making_dataset(train_err, train_user_number, train_user_id_min, train_model_nm, train_avg_err, train_time_interval, "train")
making_dataset(test_err, test_user_number, test_user_id_min, test_model_nm, test_avg_err, test_time_interval, "test")
```

```python
validation score - 0.8132780000000001
submission score - 0.8148646809
```

**각 유저별로 시간대에 에러 발생 빈도수**

```python
def hours(df, user_number, which):
    dataset = np.zeros((user_number, 24))
    df['datetime'] = df['time'].apply(return_time)
    user_hours = df.groupby('user_id')['datetime'].unique().values.tolist()
    hour_bins = df.groupby(['user_id', 'datetime'])['errtype'].count().values.tolist()
    init = 0
    for idx, id_hours in enumerate(user_hours):
        id_hours = sorted(id_hours)
        hb = hour_bins[init: init + len(id_hours)]
        init += len(id_hours)
        # 각 유저별 시간대와 해당 시간대에 에러 발생 빈도수
        if which == "test" and idx >= 13262:
            for hours, bins in zip(id_hours, hb):
                dataset[idx + 1][hours] = bins
        else:
            for hours, bins in zip(id_hours, hb):
                dataset[idx][hours] = bins

    return dataset
```

```python
validation score - 0.8141719000000001
submission score - 0.809710857	
```

**불만을 제기한 유저들과 그렇지 않은 유저들의 errcode 빈도 (상위 15개)**

```python
train_err = pd.read_csv(PATH + 'train_err_data.csv')
train_err.dropna(inplace=True)
unique_err = train_err['errcode'].unique().tolist()
train_prob = pd.read_csv(PATH + 'train_problem_data.csv')
prob_user = train_prob['user_id'].unique().tolist()
nonprob_user = train_err['user_id'].unique().tolist()

check_errcode = {}
for err in unique_err:
    check_errcode[err] = 0

for user in prob_user:
    user_errcode = train_err[train_err['user_id'] == user]['errcode'].unique().tolist()
    for err in user_errcode:
        check_errcode[err] += 1
check_errcode = sorted(check_errcode.items(), key=lambda item: item[1], reverse=True)
print(check_errcode[:15])

show_errcode_distribution(prob_user, "problem_user")
show_errcode_distribution(nonprob_user, "nonproblem_user")
```

```python 
# 불만을 나타낸 유저들의 errcode 발생 빈도 (상위 15개)
[(5, 4991), (3, 4983), (600, 4208), (771, 3961), (418, 3432), (3858, 3262), (4344, 3231), (4350, 2928), (4341, 2663), (4295, 2596), (4263, 2470), (367, 1918), (4315, 1868), (4343, 1344), (1804, 1081)]

# 불만을 나타내지 않은 유저들의 errcode 발생 빈도 (상위 15개)
[(5, 9939), (3, 9793), (600, 7122), (771, 6294), (3858, 5173), (4344, 4759), (418, 4739), (4350, 4457), (4341, 3651), (4295, 3546), (4263, 3390), (4315, 2217), (4311, 2155), (367, 2077), (4343, 1416)]
```

두 그룹에서 상위의 errcode 대부분이 유사한 발생빈도를 보임

**두 그룹간의 비율차이가 가장 큰 errcode**

```python
def show_errcode_distribution(user_id):
    check_errcode = np.zeros(4353)
    for user in user_id:
        user_errcode = train_err[train_err['user_id'] == user]['errcode'].unique().tolist()
        for err in user_errcode:
            check_errcode[err] += 1

    return check_errcode


prob_err = show_errcode_distribution(prob_user)
nonprob_err = show_errcode_distribution(nonprob_user)
sub_err = {}
for i, val in enumerate(prob_err):
    sub_err[i] = abs(val - nonprob_err[i])
sorted_sub = sorted(sub_err.items(), key=lambda item: item[1], reverse=True)

prob_val = []
nonprob_val = []
label = []
for i in range(5):
    label.append(sorted_sub[i][0])
    prob_val.append(prob_err[sorted_sub[i][0]] / 5000)
    nonprob_val.append(nonprob_err[sorted_sub[i][0]] / 10000)
x = np.arange(5)
plt.rc('font', family='Malgun Gothic')
plt.bar(x, prob_val, width=0.3, color="b", label="불만을 제기한 유저")
plt.bar(x + 0.3, nonprob_val, width=0.3, color="r", label="불만을 제기하지 않은 유저")
plt.xticks(x, label)
plt.xlabel("두 그룹에서 가장 많은 비율로 차이나는 errcode")
plt.ylabel("%")
plt.legend()
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/104999624-80487000-5a70-11eb-8bd7-b4f714f97f1a.png" width=60% />

위의 5개 errcode에서 불만을 나타낸 그룹과 그렇지 않은 그룹의 비율차이가 가장 큼

해당 에러코드의 발생빈도가 불만여부에 영향을 미친다고 판단

```python
important_errcode = [418, 367, 4344, 4341, 4295]


def important_errcode_count(df, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 5))
    for idx, ec in enumerate(important_errcode):
        sub_dataset = df[df['errcode'] == ec].groupby('user_id')['errcode'].count().reset_index().values
        for sd in sub_dataset:
            dataset[sd[0] - user_id_min][idx] = sd[1]

    dataset = pd.DataFrame(dataset, columns=important_errcode)
    dataset.to_csv("{}_important_errcode.csv".format(which), index=False)


important_errcode_count(train_err, train_user_number, train_user_id_min, "train")
important_errcode_count(test_err, test_user_number, test_user_id_min, "test")
```

불만을 제기한 유저와 그렇지 않은 경우, quality data가 존재하는 유저와 그렇지 않은 경우의 errtype 분포비교

```python
prob_user = train_prob['user_id'].unique().tolist()

def plot_errtype_dist(user_id):
    prob_dist = np.zeros(41)
    nonprob_dist = np.zeros(41)
    id_errtype = train_err.groupby('user_id')['errtype'].value_counts().index.values
    val = train_err.groupby('user_id')['errtype'].value_counts().values
    for ie, count in zip(id_errtype, val):
        if ie[0] in user_id:
            if ie[1] > 29:
                prob_dist[ie[1] - 2] += count
            else:
                prob_dist[ie[1] - 1] += count
        else:
            if ie[1] > 29:
                nonprob_dist[ie[1] - 2] += count
            else:
                nonprob_dist[ie[1] - 1] += count
	
    # 유저들의 수에 따라 나눠줌 (평균치)
    prob_dist /= 5000
    nonprob_dist /= 10000
    plt.bar(range(41), prob_dist.tolist())
    plt.show()
    plt.bar(range(41), nonprob_dist.tolist())
    plt.show()
    
plot_errtype_dist(prob_user)
```

**불만을 제기한 유저들의 errtype 분포**

<img src="https://user-images.githubusercontent.com/58063806/105651240-a7cc8c00-5ef9-11eb-8060-235d98a53541.png" width=60% />

**불만을 제기하지 않은 유저들의 errtype 분포**

<img src="C:\Users\wykim\AppData\Roaming\Typora\typora-user-images\image-20210125104035501.png" width=60% />

errtype 21, 22번에서 불만을 제기한 유저들의 빈도가 2배 이상 많은 것을 확인



```python
def plot_model_nm_dist(user_id):
    prob_dist = np.zeros(9)
    nonprob_dist = np.zeros(9)
    id_errtype = train_err.groupby('user_id')['model_nm'].value_counts().index.values
    val = train_err.groupby('user_id')['model_nm'].value_counts().values
    for ie, count in zip(id_errtype, val):
        if ie[0] in user_id:
            prob_dist[ie[1]] += count
        else:
            nonprob_dist[ie[1]] += count

    prob_dist /= 5000
    nonprob_dist /= 10000
    plt.bar(range(9), prob_dist.tolist())
    plt.show()
    plt.bar(range(9), nonprob_dist.tolist())
    plt.show()

plot_model_nm_dist(prob_user)
```

**불만을 제기한 유저들의 model_nm분포**

<img src="https://user-images.githubusercontent.com/58063806/105651714-f3cc0080-5efa-11eb-8fb7-306245aaa984.png" width=60% />

**불만을 제기하지 않은 유저들의 model_nm분포**

<img src="C:\Users\wykim\AppData\Roaming\Typora\typora-user-images\image-20210125105006641.png" width=60% />

model_nm_1에 대해서 불만의 제기한 유저들의 빈도가 2배정도 많음을 확인



<img src="https://user-images.githubusercontent.com/58063806/105681699-ce5be880-5f34-11eb-8c53-4ae5fe295838.png" width=60% />

**train_err에서 quality에 없는 user_id에 대한 errtype 분포**

<img src="https://user-images.githubusercontent.com/58063806/105681823-fcd9c380-5f34-11eb-99d9-d7ca6b44fd52.png" width=60% />

**train_err에서 quality에 있는 user_id에 대한 errtype 분포**



전체적으로 비슷한 분포지만 errtype 3에서 다른 분포를 보임

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

<img src="C:\Users\0864h\AppData\Roaming\Typora\typora-user-images\image-20210116000157967.png" width=100% />

error_time_interval 추가 

<img src="https://user-images.githubusercontent.com/58063806/104753134-ebf1bb00-579a-11eb-8246-31d80480940e.png" width=100% />

시간대에 대한 24개의 피처추가 (대부분 높은 중요도를 보임) 