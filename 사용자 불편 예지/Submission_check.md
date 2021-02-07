## Submission

#### Validation

StratifiedKFold로 5번의 검증을 진행 

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score = []
for train_idx, valid_idx in skf.split(x, y):
    train_x, valid_x = x[train_idx], x[valid_idx]
    train_y, valid_y = y[train_idx], y[valid_idx]
    evals = [(valid_x, valid_y)]
    model.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
    valid_prob = model.predict_proba(valid_x)[:, 1]
    auc_score = roc_auc_score(valid_y, valid_prob)
    score.append(auc_score)
```

대부분의 경우에서 data scaling을 진행했을때 더 낮은 score를 기록 (구성한 feature들의 값을 그대로 유지해야한다고 판단, scaling 사용X)



**유저 별 각 errtype의 빈도** 

```python
train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000

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

```python
model = XGBClassifier(random_state=42)
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

학습을 위한 모델을 결정하기 위한 submission (가장 점수가 높았던 lightgbm을 사용하기로 결정)



**유저별 각 errtype의 빈도 + 유저별 가장 많이 등장한 errcode + 해당 model 사용빈도**  

```python
train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000

sub_data = train_err[['user_id', 'errtype', 'errcode', 'model_nm']].values
dataset = np.zeros((train_user_number, 53))

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
```

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.8145382
# after standard scaling validation score - 0.813758
# submission score - 0.81732
```

AUC가 0.01 정도 상승한 결과를 보임



#### errtype 29를 제외하고 dataset 구성

**유저별 각 errtype의 빈도 + 유저별 가장 많이 등장한 errcode + 해당 model 사용빈도 + 유저별 하루동안 발생한 평균 에러의 갯수** 

```python
def making_dataset(df, user_number, user_id_min, avg_err, time_interval):
    sub_data = df[['user_id', 'errtype', 'errcode', 'model_nm']].values
    dataset = np.zeros((user_number, 52))

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
```

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.81472955
# submission score - 0.8169682162	
```



**유저별 각 errtype의 빈도 + 유저별 가장 많이 등장한 errcode + 해당 model 사용빈도 + 유저별 하루동안 발생한 평균 에러의 갯수 + 유저별 에러 발생 간격**

```python
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
```

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.8132780000000001
# submission score - 0.8148646809
```



**유저별 각 errtype의 빈도 + 유저별 가장 많이 등장한 errcode + 해당 model 사용빈도 + 유저별 하루동안 발생한 평균 에러의 갯수 + 유저별 에러 발생 간격 + 유저별 시간대(0 ~ 23)에 에러 발생 빈도**

```python
def hours(df, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 24))
    df['datetime'] = df['time'].apply(return_time)
    index = df.groupby(['user_id'])['datetime'].value_counts().index
    count = df.groupby(['user_id'])['datetime'].value_counts().values
    for idx, val in zip(index, count):
        dataset[idx[0] - user_id_min][idx[1]] = val 
    # # quality data 일때
    # for idx, val in zip(index, count):
    #     dataset[idx[0] - user_id_min][idx[1]] = val / 12

    dataset = pd.DataFrame(dataset)
    dataset.to_csv("{}_err_per_hours.csv".format(which), index=False)
```

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.8141719000000001
# submission score - 0.809710857	
```



**유저별 각 errtype의 빈도 + 유저별 가장 많이 등장한 errcode + 해당 model 사용빈도 + 유저별 하루동안 발생한 평균 에러의 갯수 + 유저별 에러 발생 간격 + 유저별 시간대(0 ~ 23)에 에러 발생 빈도 + 유저별 quality_log 발생 횟수**

```python
def cal_quality_bins(quality_log, user_number, user_id_min):
    q_user = quality_log.groupby('user_id')['time'].count().index
    q = np.array(quality_log.groupby('user_id')['time'].count().values) / 12
    dataset = np.zeros(user_number)
    for user, bin in zip(q_user, q):
        dataset[user - user_id_min] = bin
    dataset = pd.DataFrame(dataset, columns=['quality_bin'])
    dataset = pd.concat((train, dataset), axis=1, ignore_index=True)

    return dataset
```

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.8159702999999998
# submission score - 0.8124374276	
```



**유저별 각 errtype의 빈도 + 사용 model_nm 빈도 + 유저별 사용 fwver 빈도 + 유저별 하루동안 발생한 평균 에러의 갯수 +유저별 에러 발생 간격 + 유저별 quality log 발생 빈도** 

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.81776885
# submission score - 0.8123721779
```



```python
def most_appear_max_val(df, user_number, user_id_min, which):
    for i in range(13):
        df['quality_{}'.format(i)].replace(',', '', regex=True, inplace=True)
        df['quality_{}'.format(i)] = df['quality_{}'.format(i)].astype(int)
    dataset = np.zeros((user_number, 13))
    dataset2 = np.zeros((user_number, 13))
    user_id = df['user_id'].unique()
    for i in range(13):
        for id in user_id:
            dataset[id - user_id_min][i] = df[df['user_id'] == id]['quality_{}'.format(i)].mode()[0]
            dataset2[id - user_id_min][i] = df[df['user_id'] == id]['quality_{}'.format(i)].max()
    col = df.columns[3:]
    most_appear = []
    max_value = []
    for name in col:
        most_appear.append("most_appear_" + name)
        max_value.append("max_value_" + name)
    dataset = pd.DataFrame(dataset, columns=most_appear)
    dataset2 = pd.DataFrame(dataset2, columns=max_value)
    dataset.to_csv("{}_most_appear_value.csv".format(which), index=False)
    dataset2.to_csv("{}_max_value.csv".format(which), index=False)
```

**유저별 errtype의 빈도 + 사용 model_nm 체크(0, 1) + 유저별 사용 fwver 체크(0, 1) + 유저별 하루동안 발생한 평균 에러의 갯수 +유저별 에러 발생 간격 + 유저별 quality log 발생 빈도 +  유저별 quality_log 별 최대값**

```python
model = LGBMClassifier(random_state=42)
# validation score - 0.81743305
# submission score - 0.8174800017	
```



```python
def count_fwver(df, df2, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 47))
    dataset2 = np.zeros((user_number, 47))
    user_fwver = df.groupby('user_id')['fwver'].value_counts().index
    for id_ver in user_fwver:
        dataset[id_ver[0] - user_id_min][id_ver[1]] = 1

    user_fwver2 = df2.groupby('user_id')['fwver'].value_counts().index
    cf2 = df2.groupby('user_id')['fwver'].value_counts().values
    for id_ver, val in zip(user_fwver2, cf2):
        dataset2[id_ver[0] - user_id_min][id_ver[1]] += val / 12

    dataset = pd.DataFrame(dataset)
    dataset.to_csv('{}_count_fwver.csv'.format(which), index=False)
    dataset2 = pd.DataFrame(dataset2)
    dataset2.to_csv('{}_quality_fwver.csv'.format(which), index=False)
week = ['Mon', 'Tue', 'Wen', 'Thu', 'Fri', 'Sat', 'Sun']

def week_bin(df, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 7))
    df['datetime'] = df['time'].apply(make_datetime)
    # 0 ~ 6, 월 ~ 일
    df['datetime'] = df['datetime'].apply(lambda x: x.weekday())
    index = df.groupby('user_id')['datetime'].value_counts().index
    count = df.groupby('user_id')['datetime'].value_counts().values
    for idx, val in zip(index, count):
        dataset[idx[0] - user_id_min][idx[1]] = val
    # # quality data 일때
    # for idx, val in zip(index, count):
    #     dataset[idx[0] - user_id_min][idx[1]] = val / 12
    dataset = pd.DataFrame(dataset)
    dataset.to_csv("{}_week_bin.csv".format(which), index=False)
```

**유저별 errtype 발생빈도 + 사용 model_nm 체크(0, 1) + 유저별 사용 fwver 체크(0, 1) +  유저별 하루동안 발생한 평균 에러의 갯수 + 유저별 에러 발생 간격 + 유저별 quality log 발생 빈도 + 유저별 사용 fwver 빈도(quality dataset) + 중요 errcode발생 빈도 + 유저별 시간대(0 ~ 23)에 err 발생 빈도 + 유저별 해당 요일 err 발생 빈도 + 유저별 해당 시간대 quality_log 발생 빈도 + 유저별 해당 요일 quality_log 발생 빈도 + 유저별 quality_log 별 가장 최빈값, 최대값** 

```python
model = LGBMClassifier(random_state=42)
# shape - (15000, 242)
# validation score - 0.81907405
# submission score - 0.8157693204
```

총 242개의 변수로 구성된 dataset 구성

