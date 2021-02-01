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
```

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
```

결측값이 있는 행을 확인

```python
print(train_err[train_err['errcode'].isnull() == True])
print(test_err[test_err['errcode'].isnull() == True])
```

<img src="https://user-images.githubusercontent.com/58063806/106231774-cd2d0300-6235-11eb-944b-d032f14a302e.png" width=70% />

```python
print(train_err[(train_err['user_id'] == 13639) & (train_err['errtype'] == 5)])
```

<img src="https://user-images.githubusercontent.com/58063806/106232272-f26e4100-6236-11eb-8ffc-fa9f08826c34.png" width=70% />

```python
print(test_err[((test_err['user_id'] == 30820) | (test_err['user_id'] == 33681) | (test_err['user_id'] == 38991) | (test_err['user_id'] == 39894)) & (test_err['errtype'] == 5)])
```

<img src="https://user-images.githubusercontent.com/58063806/106232542-9c4dcd80-6237-11eb-8a9f-f822729da7f6.png" width=70%/>

<img src="https://user-images.githubusercontent.com/58063806/106232793-3ada2e80-6238-11eb-8833-dc7ea3229a0b.png" width=70% />

<img src="https://user-images.githubusercontent.com/58063806/106232855-60ffce80-6238-11eb-85df-c05e2454c49f.png" width=70% />

<img src="https://user-images.githubusercontent.com/58063806/106232894-7ecd3380-6238-11eb-9210-7b94502adf95.png" width=70% />

결측값이 있는 행과 errcode를 제외하고 모두 동일한 행을 발견 

결측값을 삭제하고 같은 errcode가 있는 행을 확인

```python
train_err.dropna(inplace=True)
test_err.dropna(inplace=True)
print(train_err[train_err['errcode'] == "40013"])
print(test_err[(test_err['errcode'] == "40053") | (test_err['errcode'] == "-1010")])
```

동일한 errcode를 가진 행은 없었음



똑같은 값이 중복되서 들어온 것으로 보아 연결이 끊어지거나 재부팅이 발생하는 경우로 판단 

해당 유저의 errcode 발생 현황

```python
print(train_err[(train_err['user_id'] == 13639) & (train_err['errtype'] == 5)].groupby('errcode').count())
```

<img src="https://user-images.githubusercontent.com/58063806/106236364-0d917e80-6240-11eb-968a-f36b3e79db67.png" width=50%/>

```python
print(test_err[((test_err['user_id'] == 30820) | (test_err['user_id'] == 33681)
                | (test_err['user_id'] == 38991) | (test_err['user_id'] == 39894))
               & (test_err['errtype'] == 5)].groupby(['user_id', 'errcode']).count())
```

<img src="https://user-images.githubusercontent.com/58063806/106237443-34e94b00-6242-11eb-80ac-7579c810196c.png" width=50%/>

공통적으로 B-A8002 errcode가 많이 발생한 것을 볼 수 있음



train_err에서 B-A8002 errcode가 발생한 유저의 수

```python
print(len(train_err[train_err['errcode'] == 'B-A8002']['user_id'].unique()))
# 5860
```

불만을 제기한 유저의 수와 해당 유저 중 B-A8002 errcode가 발생한 유저의 수 

```python
prob_id = train_prob['user_id'].unique()
selected_user = train_err[train_err['errcode'] == 'B-A8002']['user_id'].unique()
count = 0
print(len(prob_id))
# 5000
for id in selected_user:
    if id in prob_id:
        count += 1
print(count)
# 2470
```

- **B-A8002 errcode가 발생한 유저 중 약 40%**에 해당하는 인원이 **불만을 제기** 

- **불만을 제기한 유저 중 거의 절반**에 해당하는 인원이 **B-A8002 errcode가 발생**한 것을 볼 수 있음

**train_quality_data**

- 시스템 퀄리티 로그
- 12행 단위로 시간이 변함(2시간 간격)

```python
train_quality = pd.read_csv(PATH+'train_quality_data.csv')
print(train_quality.isnull().sum())
```

<img src="https://user-images.githubusercontent.com/58063806/104149828-151cef00-541b-11eb-9dc7-949d51ef1c41.png" width=20% />

여러 피처에서 상당히 많은 양의 결측값이 관측

**test_quality_data**

```python
print(test_quality.isnull().sum())
```

<img src="https://user-images.githubusercontent.com/58063806/104280404-33ecb580-54ef-11eb-91e8-d8965ffc2ada.png" width=20% />

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



결측값이 있는 quality log 별로 fwver을 체크

```python
print(train_quality[(train_quality['quality_0'].isnull() == True)]['fwver'].unique())
print(train_quality[(train_quality['quality_2'].isnull() == True)]['fwver'].unique())
print(train_quality[(train_quality['quality_5'].isnull() == True)]['fwver'].unique())
```

<img src="https://user-images.githubusercontent.com/58063806/106240708-14bc8a80-6248-11eb-8882-a222f4818edc.png" width=70% />

가장 많은 결측값이 있는 quality_0은 03.11.1149, 03.11.1167을 제외한 나머지 fwver은 quality_2의 fwver과 일치

```python
# fwver이 10이고 quality_0 or quality_2에서 결측값이 관찰된 행
print(train_quality[(train_quality['fwver'] == '10') &
                    (train_quality['quality_0'].isnull() == True | train_quality['quality_2'].isnull() == True)])

# fwver이 10이고 quality_0 and quality_2에서 결측값이 관찰된 행
print(train_quality[(train_quality['fwver'] == '10') &
                    ((train_quality['quality_0'].isnull() == True) & (train_quality['quality_2'].isnull() == True))])
```

위의 결과 두개가 일치하는 것을 발견 (해당 fwver에서는 두 quality log가 유의미한 관계를 가짐)

나머지 fwver에서도 일치하는 것을 확인

```python
print(train_quality[(train_quality['fwver'] == '04.22.1750') &
                    ((train_quality['quality_0'].isnull() == True) | (train_quality['quality_2'].isnull() == True) | (train_quality['quality_5'].isnull() == True))])

print(train_quality[(train_quality['fwver'] == '04.22.1750') &
                    ((train_quality['quality_0'].isnull() == True) & (train_quality['quality_2'].isnull() == True) & (train_quality['quality_5'].isnull() == True))])
```

세 개의 quality log에 대해서 확인

세 quality log 동시에 결측값을 갖는 경우는 없음을 확인 



전체 dataset에서 quality_0과 quality_2가 같은 경우 확인

```python
print(train_quality[(train_quality['quality_0'] == train_quality['quality_2'])])
```

약 83만개의 행 중에서 68만개가 넘는 경우에 같음을 확인

**quality_0, 2가 동시에 결측값을 갖는 경우는 같은 값을 갖는다고 가정**



```python
train_quality['quality_5'].fillna(method="pad", inplace=True)
```

quality_5는 전 row의 값으로 결측값을 대체

```python
print(train_quality[(train_quality['quality_0'] == train_quality['quality_1']) & (train_quality['quality_1'] == train_quality['quality_2'])][['quality_0', 'quality_1', 'quality_2']].value_counts())
```

<img src="https://user-images.githubusercontent.com/58063806/106295237-866df600-6293-11eb-9b80-89158ceaac23.png" width=50% />

약 83만개의 행 중 약 77만개의 행에서 quality_0, 1, 2가 동일(대부분 값은 0, -1)

```python
def fill_quality_na(df, which):
    q0_fwver = df[(df['quality_0'].isnull() == True)]['fwver'].unique()
    q2_fwver = df[(df['quality_2'].isnull() == True)]['fwver'].unique()
    for ver in q0_fwver:
        index = df.loc[(df['fwver'] == ver)
                                  & (df['quality_0'].isnull() == True) & (df['quality_2'].notnull() == True)].index
        if len(index) > 0:
            df.loc[index, "quality_0"] = df.loc[index, "quality_2"]
    for ver in q2_fwver:
        index = df.loc[(df['fwver'] == ver)
                       & (df['quality_2'].isnull() == True) & (df['quality_0'].notnull() == True)].index
        if len(index) > 0:
            df.loc[index, "quality_2"] = df.loc[index, "quality_0"]
    
    # 이때는 quality_0과 2가 동일한 row에서 결측값을 가짐
    # quality_1의 값이 -1인 경우에 quality_0, 2의 결측값을 -1로
    index = df.loc[(df['quality_0'].isnull() == True) & (df['quality_1'] == -1)].index
    df.loc[index, "quality_0"] = df.loc[index, "quality_2"] = df.loc[index, "quality_1"]
    # 나머지는 결측값 0으로 대체
    df.fillna(0, inplace=True)
    df.to_csv("filled_{}_quality.csv".format(which), index=False)
```



```python
print(test_quality[(test_quality['quality_0'].isnull() == True)]['fwver'].unique())
print(test_quality[(test_quality['quality_1'].isnull() == True)]['fwver'].unique())
print(test_quality[(test_quality['quality_2'].isnull() == True)]['fwver'].unique())
print(test_quality[(test_quality['quality_5'].isnull() == True)]['fwver'].unique())
```

<img src="https://user-images.githubusercontent.com/58063806/106240933-7e3c9900-6248-11eb-8fb5-5cfa3fe9fb83.png" width=70%/>

fwver 10이나 8.5.3과 비슷한 양상을 보임

nan은 user_id가 43262인 유저에 대한 정보 

```python
print(test_quality[(test_quality['fwver'].isnull() == True)])
```

<img src="https://user-images.githubusercontent.com/58063806/106298974-f54d4e00-6297-11eb-88d7-be77d305e373.png" width=100% />

결과를 보면 user_id 43262의 모든 행에서 quality_0, 2 둘 다 결측값을 갖는 것을 알 수 있음

```python
print(test_quality[(test_quality['quality_0'].isnull() == True) & (test_quality['quality_2'].isnull() == True)]['fwver'].unique())
# ['10' '8.5.3' nan]
test_quality['fwver'].fillna('8.5.3', inplace=True)
```

fwver 10과 8.5.3 중 8.5.3의 형태와 조금 더 유사한 것을 확인하고 fwver의 결측값을 8.5.3으로 대체



```python
test_quality['quality_5'].fillna(method="pad", inplace=True)
```

train과 마찬가지로 quality_5는 해당 row의 전 값으로 대체

```python
print(test_quality[(test_quality['quality_0'] == test_quality['quality_1']) & (test_quality['quality_1'] == test_quality['quality_2'])][['quality_0', 'quality_1', 'quality_2']].value_counts())
```

<img src="https://user-images.githubusercontent.com/58063806/106346806-0bd9c080-62fd-11eb-86d1-c9ba5f8eec4f.png" width=50% />

약 75만개의 행 중 60만개의 행에서 quality_0, 1, 2가 동일 (값은 대부분 0, -1)

quality_0, 2가 -1인 경우를 제외하고는 0으로 대체



```python
print(test_quality[(test_quality['quality_0'] == test_quality['quality_2'])])
```

약 75만개의 행 중 64만개의 행에서 quality_0과 2가 동일

train_quality와 동일하게 결측값 처리



**train_problem_data**

- 사용자의 불만이 접수된 시간
- 불만이 접수된 시간 이후에도 train_err_data를 보면 에러 로그는 계속 발생했음을 알 수 있음



불만을 제기한 유저들의 errcode 파악

```python
all_errcode = list(set().union(train_err['errcode'], test_err['errcode']))

def show_errcode(df):
    Dict = dict(zip(all_errcode, [0] * len(all_errcode)))
    print(Dict)
    prob_user = train_prob['user_id'].unique()

    for user in prob_user:
        index = df[df['user_id'] == user]['errcode'].value_counts().index
        count = df[df['user_id'] == user]['errcode'].value_counts().values
        for idx, val in zip(index, count):
            Dict[idx] += val

    Dict = sorted(Dict.items(), key=lambda item: item[1], reverse=True)
    print(Dict[:100])
```

**train**

<img src="https://user-images.githubusercontent.com/58063806/106349811-51ee4e80-6314-11eb-92b7-4b6ac75fa8a1.png" width=100% />

B-A8002는 물론 connection, 연결과 관련된 errcode들이 많이 발생한 것을 알 수 있음



불만을 제기한 유저들의 fwver 파악

```python
all_fwver = list(set().union(train_err['fwver'].unique(), test_err['fwver'].unique()))

def show_fwver(df):
    Dict = dict(zip(all_fwver, [0] * len(all_fwver)))
    prob_user = train_prob['user_id'].unique()

    for user in prob_user:
        index = df[df['user_id'] == user]['fwver'].value_counts().index
        count = df[df['user_id'] == user]['fwver'].value_counts().values

        for idx, val in zip(index, count):
            Dict[idx] += val

    Dict = sorted(Dict.items(), key=lambda item: item[1], reverse=True)
    print(Dict)
```

**train_err**

<img src="https://user-images.githubusercontent.com/58063806/106350031-1e142880-6316-11eb-9a79-ad529d7e6207.png" width=100% />

**train_quality**

<img src="https://user-images.githubusercontent.com/58063806/106384979-32891b80-6411-11eb-8983-e1d43995bc20.png" width=100% />



해당 fwver을 사용하는 유저들의 불만 제기확률

```python
fwver = ['04.16.3553', '04.22.1750', '04.33.1261', '03.11.1167', '04.22.1778', '05.15.2138', '04.33.1185', '04.16.3571']
for ver in fwver:
    tq = train_quality[train_quality['fwver'] == ver]['user_id'].unique()
    # te = train_err[train_err['fwver'] == ver]['user_id'].unique()
    tp = train_prob['user_id'].unique()
    count = 0
    for id in tq:
        if id in tp:
            count += 1
    print(ver, count, count / len(tq))

# train_quality
# 04.16.3553 667 0.4622314622314622
# 04.22.1750 868 0.4412811387900356
# 04.33.1261 851 0.4359631147540984
# 03.11.1167 111 0.21595330739299612
# 04.22.1778 513 0.46636363636363637
# 05.15.2138 487 0.28248259860788866
# 04.33.1185 476 0.4103448275862069
# 04.16.3571 43 0.6323529411764706

# train_err
# 04.16.3553 1314 0.46316531547409234
# 04.22.1750 1630 0.37635649965365964
# 04.33.1261 1739 0.39105014616595457
# 03.11.1167 141 0.20644216691068815
# 04.22.1778 1586 0.3799712505989459
# 05.15.2138 720 0.23233301064859632
# 04.33.1185 1128 0.32828870779976715
# 04.16.3571 295 0.5876494023904383
```



상위 5개의 fwver의 errcode를 확인

```python
fwver = ['04.16.3553', '04.22.1750', '04.33.1261', '03.11.1167', '04.22.1778']

for ver in fwver:
    print(train_err[train_err['fwver'] == ver]['errcode'].value_counts().head(50))
```

connection timeout 등 connection과 관련된 error, B-A8002,  NFANDROID2, S-61001, active, standby 등의 error가 많이 발생

test_err에서도 비슷한 양상을 보임



중요하다고 생각되는 errcode들의 각 유저별 발생 빈도

```python
imp_errcode = ["connection", "B-A8002", "S-", "NFANDROID2", "active", "standby", "scanning timeout"]


def add_imp_errcode(df, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 7))
    index = df.groupby('user_id')['errcode'].value_counts().index
    count = df.groupby('user_id')['errcode'].value_counts().values
    for idx, val in zip(index, count):
        # connection 키워드가 포함된 errcode
        if "connection" in idx[1]:
            dataset[idx[0] - user_id_min][0] += val
        # S- 키워드가 포함된 errcode
        elif "S-" in idx[1]:
            dataset[idx[0] - user_id_min][2] += val
        if idx[1] in imp_errcode:
            dataset[idx[0] - user_id_min][imp_errcode.index(idx[1])] += val
    dataset = pd.DataFrame(dataset, columns=imp_errcode)
    dataset.to_csv("{}_important_errcode.csv".format(which), index=False)
```



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



평균적으로 하루에 발생한 에러 데이터를 추가

 ```python
# validation score - 0.81472955
# submission score - 0.8169682162	
 ```

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
validation score - 0.8141719000000001
submission score - 0.809710857	
```



**각 유저별 해당 요일에 발생한 에러빈도**

```python
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

errtype 22, 23번에서 불만을 제기한 유저들의 빈도가 2배 이상 많은 것을 확인



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

전체적으로 비슷한 분포지만 errtype 4에서 다른 분포를 보임



**quality log가 발생하기 직전 발생한 errtype의 분포**

<img src="https://user-images.githubusercontent.com/58063806/105792647-188fa900-5fcb-11eb-8520-2e65987e936d.png" width=60% />

errtype 4, 15, 31이 많이 발생



**quality log가 발생하기 직전 발생한 errcode의 수**

<img src="https://user-images.githubusercontent.com/58063806/106384521-9eb65000-640e-11eb-91f9-79c7bfedbd66.png" width=100% />

**각 user별 quality log 발생 횟수**

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

# validation score - 0.8159702999999998
# submission score - 0.8124374276	

# error_time_interval 제거
# validation score - 0.81577165
# submission score - 0.815548357	

# error_time_interval, errcode, errors_per_day 제거
# validation score - 0.8168468500000001
# submission score - 0.8152312361	
```

**err, quality data 간의 fwver 분포비교**

```python
err = train_err['fwver'].value_counts().reset_index()
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].barh(err['index'], err['fwver'])
axes[0].set_title("train_err_fwver")
axes[0].set_yticklabels(err['index'])
train_quality = pd.read_csv('filled_train_quality.csv')
qual = train_quality['fwver'].value_counts().reset_index()
axes[1].barh(qual['index'], qual['fwver'])
axes[1].set_title("train_quality_fwver")
axes[1].set_yticklabels(qual['index'])
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/105932316-12afcb80-6090-11eb-8a28-783f4ff33ce0.png" width=90% />

비교를 해보면 상위 6개에 해당하는 fwver는 순서는 조금씩 달라도 err와 quality에서 일치

**fwver - 8.5.3, 10의 경우에는 err에 비해 quality에서 매우 높게 나타나고있음**  

해당 firmware version에서 시스템 작동 상태 중 관련보다 문제가 많이 발생한다는 것을 의미 



**각 user별 fwver 체크**

```python
def count_fwver(df, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 47))
    user_fwver = df.groupby('user_id')['fwver'].value_counts().index
    for id_ver in user_fwver:
        dataset[id_ver[0] - user_id_min][id_ver[1]] = 1  # true

    dataset = pd.DataFrame(dataset)
    dataset.to_csv('{}_count_fwver.csv'.format(which), index=False)
```



유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling)

```python
# shape - (15000, 53)
# validation score - 0.8168471
```

유저별 errtype 발생빈도와 + 유저별 사용 fwver + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling)

```python
# shape - (15000, 91)
# validation score - 0.8162563
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver 

``` python
# shape - (15000, 100)
# validation score - 0.81776885
# submission score - 0.8123721779

# feature_importance가 0인 피처들을 제거
# shape - (15000, 61)
# validation score - 0.8175133499999999
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver + 시간대 별 err 발생 횟수

```python
# shape - (15000, 124)
# validation score - 0.8138381000000001
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver +  중요 errcode 발생 빈도

```python
# shape - (15000, 107)
# validation score - 0.81603875
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver +  중요 errcode 발생 빈도 + 유저별 해당 시간대 err 발생 빈도

```python
# shape - (15000, 131)
# validation score - 0.8167112
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver +  중요 errcode 발생 빈도 + 유저별 해당 시간대 err 발생 빈도 + 유저별 해당 요일 err 발생 빈도

```python
# shape - (15000, 138)
# validation score - 0.816304
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver +  중요 errcode 발생 빈도 + 유저별 해당 시간대 err 발생 빈도 + 유저별 해당 요일 err 발생 빈도 + 유저별 quality_log 별 가장 많이 발생한 값, 최대값 

```python
# shape - (15000, 164)
# validation score - 0.8161432499999999
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver +  중요 errcode 발생 빈도 +  유저별 quality_log 별 최대값

```python
# shape - (15000, 120)
# validation score - 0.81743305
# submission score - 0.8174800017	
```



**각 유저별 quality log의 평균값**

```python
def mean_quality(df, user_number, user_id_min, which):
    dataset = np.zeros((user_number, 13))
    for i in range(13):
        user = df.groupby('user_id')['quality_{}'.format(i)].mean().index
        val = df.groupby('user_id')['quality_{}'.format(i)].mean().values
        for id, mv in zip(user, val):
            dataset[id - user_id_min][i] = mv

    dataset = pd.DataFrame(dataset, columns=df.columns[3:])
    dataset.to_csv("{}_mean_quality.csv".format(which), index=False)
```

train, test 모두 quality_3, quality_4는 모든 값이 0으로 구성

 

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver  + 유저별 quality 평균값

```python
# shape - (15000, 113)
# validation score - 0.8167156500000001
```



**각 fwver별로 quality_log의 값이 0이 아닌 분포**

```python
fwver = list(set().union(train_quality['fwver'].unique(), test_quality['fwver'].unique()))
scaler = MinMaxScaler()
global check
check = False


def fwver_quality_log(df, which):
    global check
    fig = plt.figure(figsize=(60, 40))
    Dict = dict(zip(fwver, range(len(fwver))))
    for i in range(13):
        index = df[(df['quality_{}'.format(i)] != 0) & (df['quality_{}'.format(i)] != '0')].groupby('fwver')[
            'quality_{}'.format(i)].count().index.tolist()
        value = df[(df['quality_{}'.format(i)] != 0) & (df['quality_{}'.format(i)] != '0')].groupby('fwver')[
            'quality_{}'.format(i)].count().values
        fw_index = df.groupby('fwver')['quality_{}'.format(i)].count().index
        fw_count = df.groupby('fwver')['quality_{}'.format(i)].count().values
        count_dict = dict(zip(fw_index, fw_count))
        if len(value) == 0:
            for idx in fwver:
                Dict[idx] = 0
        else:
            for idx, val in zip(index, value):
                Dict[idx] = val / count_dict[idx]
            for idx in set(fwver) - set(index):
                Dict[idx] = 0
        values = np.array(list(Dict.values()))
        values = values.reshape(-1, 1)
        if not check:
            values = scaler.fit_transform(values)
            check = True
        else:
            values = scaler.transform(values)
        values = np.squeeze(values, axis=1)
        ax = fig.add_subplot(3, 5, i + 1)
        ax.bar(Dict.keys(), values)
        ax.set_title('quality_{}_fwver_bins'.format(i))
        ax.set_xticklabels(list(Dict.keys()), rotation=90)
    plt.savefig("plot/{}_avg_plot.png".format(which))
```

quality_0, 2

quality_1, 5, 6, 10, 11

위의 log들의 분포가 유사

```python
err = set().union(train_err['fwver'].unique(), test_err['fwver'].unique())
qual = set().union(train_quality['fwver'].unique(), test_quality['fwver'].unique())
print(len(err - qual), err - qual)
print(len(qual - err), qual - err)

11 {'04.22.1170', '04.33.1171', '04.33.1095', '05.15.2090', '04.82.1730', '04.16.2641', '10.22.1780', '04.73.2569', '04.16.3569', '04.73.2577', '05.15.2092'}
1 {'09.17.1431'}
```

fwver 09.17.1431는 err data에는 없고 quality data에만 있는 것을 발견



quality dataset에서 fwver 09.17.1431 사용하는 유저들의 불만 현황

```python
selected_user = train_quality[train_quality['fwver'] == "09.17.1431"]['user_id'].unique()
prob_id = train_prob['user_id'].unique()
for id in selected_user:
    if id in prob_id:
        print(id, ": problem")
    else:
        print(id, ": nonproblem")
```

이 결과 **25명의 유저 중 24명의 유저가 실제로 불만을 제기함**

(fwver 09.17.1431을 사용하는 유저들을 대부분 불만을 제기한다고 유추 가능)



**err dataset은 fwver 사용 유무, quality dataset은 fwver 빈도 체크**

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
```

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver 체크(err dataset) +  유저별 사용 fwver 빈도(quality dataset) + 중요 errcode발생 빈도 + 유저별 해당 시간대 err 발생 빈도 + 유저별 해당 요일 err 발생 빈도 + 유저별 해당 시간대 quality_log 발생 빈도 + 유저별 해당 요일 quality_log 발생 빈도 + 유저별 quality_log 별 가장 많이 발생한 값, 최대값 

```python
train = pd.concat((train, train_errcode, train_quality_fwver, train_quality_hour, train_quality_week, train_hour, train_week, train_quality_most, train_quality_max), axis=1, ignore_index=True)
# shape - (15000, 242)
# validation score - 0.81907405
# submission score - 0.8157693204
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

<img src="C:\Users\0864h\AppData\Roaming\Typora\typora-user-images\image-20210116000157967.png" width=100% />

error_time_interval 추가 

<img src="https://user-images.githubusercontent.com/58063806/104753134-ebf1bb00-579a-11eb-8246-31d80480940e.png" width=100% />

시간대에 대한 24개의 피처추가 (대부분 높은 중요도를 보임) 

<img src="https://user-images.githubusercontent.com/58063806/105999457-10318e00-60f1-11eb-8e46-9af3dc217a06.png" width=100% />

유저별 errtype 발생빈도와 + 사용 model_nm + 유저별 quality log 발생 빈도 + 하루 평균 err 발생량 + 에러 간의 interval(Minmax scaling) + 유저별 사용 fwver 

상당수의 피처 중요도가 0으로 나타남