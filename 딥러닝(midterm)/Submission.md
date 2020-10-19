## Submission

#### 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
Train = pd.read_csv("dataset/trainset.csv")
Test = pd.read_csv("dataset/testset.csv")
pd.set_option('display.max_rows', 1000)
# 최대 열 수 설정
pd.set_option('display.max_columns', 1000)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)
# print(Train.isnull().sum()) # 결측값 확인
# print(Train.value_counts("class"))  # 클래스별 갯수확인
# print(Train.groupby("class").mean())
x = Train.drop(columns="class")
y = Train["class"]
df = pd.concat([x, Test], axis=0)
corr = x.corr(method="pearson")
c = x.columns.tolist()
for i in range(0, corr.shape[1]):
    filtering = corr.iloc[[i], :] >= 0.8
    select = np.where(filtering == True)[1]
    new_f = df.iloc[:, select]
    df['corr {}_mean'.format(c[i])] = new_f.mean(axis=1)
    df['corr {}_var'.format(c[i])] = new_f.var(axis=1)
    df['corr {}_std'.format(c[i])] = new_f.std(axis=1)
    df['corr {}_maxmin'.format(c[i])] = new_f.max(axis=1) - new_f.min(axis=1)
c = df.columns.tolist()
x = df[:4280]
Test = df[4280:]
scaler = StandardScaler()
x = scaler.fit_transform(x)
Test = scaler.transform(Test)
x = pd.DataFrame(data=x, columns=c)
x = x[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'corr c1_mean', 'corr c1_var', 'corr c1_std', 'corr c2_mean', 'corr c2_var', 'corr c2_std', 'corr c3_mean', 'corr c3_var', 'corr c3_std', 'corr c4_mean', 'corr c4_var', 'corr c4_std', 'corr c5_mean', 'corr c5_var', 'corr c5_std', 'corr c6_mean', 'corr c6_var', 'corr c6_std', 'corr c7_mean', 'corr c7_var', 'corr c7_std', 'corr c8_mean', 'corr c8_var', 'corr c8_std', 'corr c9_mean', 'corr c9_var', 'corr c10_mean', 'corr c10_var', 'corr c10_std', 'corr c11_mean', 'corr c11_var', 'corr c11_std', 'corr c12_mean', 'corr c12_var', 'corr c13_mean', 'corr c13_var', 'corr c13_std', 'corr c14_mean', 'corr c14_var', 'corr c14_std', 'corr c15_mean', 'corr c15_var', 'corr c15_std', 'corr c16_mean', 'corr c16_var', 'corr c16_std', 'corr c17_mean', 'corr c17_var', 'corr c17_std', 'corr c18_mean', 'corr c18_var', 'corr c18_std', 'corr c19_mean', 'corr c19_var', 'corr c19_std', 'corr c20_mean', 'corr c20_var', 'corr c20_std', 'corr c21_mean', 'corr c21_var', 'corr c21_std', 'corr c22_mean', 'corr c22_var', 'corr c22_std', 'corr c23_mean', 'corr c23_var', 'corr c23_std', 'corr c24_mean', 'corr c24_var', 'corr c24_std', 'corr c25_mean', 'corr c25_var', 'corr c25_std', 'corr c26_mean', 'corr c26_var', 'corr c26_std', 'corr c27_mean', 'corr c27_var', 'corr c27_std', 'corr c28_mean', 'corr c28_var', 'corr c28_std', 'corr c29_mean', 'corr c29_var', 'corr c29_std', 'corr c30_mean', 'corr c30_var', 'corr c30_std', 'corr c31_mean', 'corr c31_var']]
model = XGBClassifier(tree_method='gpu_hist', n_estimators=200, n_jobs=-1, learning_rate=0.25, subsample=0.7, max_depth=9, objective="multi:softprob", random_state=42)
res = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
i = 0
ans = np.zeros([1833, 6])
acc = np.zeros(10)
for train_idx, test_idx in skf.split(selected_x, y):
    train_x, test_x = selected_x.iloc[train_idx], selected_x.iloc[test_idx]
    train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
    evals = [(test_x, test_y)]
    model.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
    accuracy = model.score(test_x, test_y)
    y_pred = model.predict_proba(Test)
    ans[i] = y_pred
    acc[i] = accuracy
    i += 1
    
ans = ans / 10
data = np.argmax(ans, axis=1)
# 숫자를 다시 문자 label로
result = []
for val in data:
    result.append(label[val])
submission = pd.DataFrame(data=result)
submission.to_csv('32163711_이현수.csv', index=True)
```

cross_validation 결과 0.9238317757009347의 정확도가 나왔지만 실제 제출결과는 0.915 정도가 나옴

검증을 마치고 해당 피처들(전체 Train 데이터)과 model로 학습을 진행결과 **0.920305676855895** 의 제출결과가 나옴 (검증 결과보다 0.003정도 작음)

#### 2

```python
# 기존의 피처에서 RFE로 얻어낸 23개의 피처로 진행해봄
['c3', 'c4', 'c5', 'c8', 'c10', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c31']

model = XGBClassifier(tree_method='gpu_hist', n_estimators=200, n_jobs=-1, learning_rate=0.25, subsample=0.7, max_depth=9, objective="multi:softprob", random_state=42)
```

cross_validation 결과 0.9182242990654206의 정확도가 나옴

실제 제출결과는 **0.920305676855895**로 동일한 결과가 나옴

#### 3

```python
# 피처 전체를 대상으로 데이터의 크기가 크지 않으므로 max_depth를 크게 설정해서 진행
model = XGBClassifier(tree_method='gpu_hist', n_estimators=200, n_jobs=-1, learning_rate=0.25, subsample=0.7, max_depth=27, objective="multi:softprob", random_state=42)

```

cross_validation결과 0.9172897196261681의 정확도가 나옴

실제 제출결과는 **0.922489082969432**로 acc가 상승한 결과가 나옴  

#### 4

```python
df = pd.concat([x, Test], axis=0)
for i in range(0, corr.shape[1]):
    filtering = (corr.iloc[[i], :]) >= 0.77
    select = np.where(filtering == True)[1]
    new_f = df.iloc[:, select]
    df['corr {}_mean'.format(c[i])] = new_f.mean(axis=1)
    df['corr {}_var'.format(c[i])] = new_f.var(axis=1)
    df['corr {}_std'.format(c[i])] = new_f.std(axis=1)
    df['corr {}_sum'.format(c[i])] = new_f.sum(axis=1)
    df['corr {}_median'.format(c[i])] = new_f.median(axis=1)
    df['corr {}_maxmin'.format(c[i])] = new_f.max(axis=1) - new_f.min(axis=1)
# 위의 코드로 파생변수를 생성하고 총 217개의 피처를 대상으로 진행
filtering = (corr.iloc[[i], :]) >= 0.77
# 상관관계 값을 조금 더 낮췄을 때 acc가 좀 더 잘나옴
model = XGBClassifier(tree_method='gpu_hist', n_estimators=200, n_jobs=-1, learning_rate=0.25, subsample=0.7, max_depth=9, objective="multi:softprob", random_state=42)
```

cross_validation 결과 0.9212616822429908의 정확도가 나옴

실제 제출결과는 **0.924672489082969**로 acc가 상승한 결과가 나옴 

#### 5(SMOTE 적용)

```python
smote = SMOTE(k_neighbors=1, random_state=42)
train_x, train_y = smote.fit_sample(train_x, train_y)
# oversampling 기법인 smote를 적용
model = XGBClassifier(tree_method='gpu_hist', n_estimators=300, n_jobs=-1, learning_rate=0.25, subsample=0.7, max_depth=9, objective="multi:softprob", random_state=42)

average accuary : 0.9196261682242991  std : 0.00986855704763686
# acc값이 약간 떨어짐

model = XGBClassifier(tree_method='gpu_hist', n_estimators=300, n_jobs=-1, learning_rate=0.35, subsample=0.7, max_depth=9, objective="multi:softprob", random_state=42)
# learning_rate를 올리면 acc값이 많이 상승
average accuary : 0.9259345794392523  std : 0.008362397274847314
        
제출 결과 : acc - 0.924672489082969
# oversampling 전의 결과와 차이가 없음

model = XGBClassifier(tree_method='gpu_hist', n_estimators=300, n_jobs=-1, learning_rate=0.25, subsample=0.7, max_depth=12, objective="multi:softprob", random_state=42)
# learning_rate를 그대로, max_depth를 증가시켜봄, 역시 acc가 상승
average accuary : 0.9247663551401869  std : 0.010490628187216662

제출 결과 : acc - 0.924672489082969
# 역시 동일 oversampling은 큰 의미가 없는 것으로 판단
```

#### 6

```python
for i in range(0, corr.shape[1]):
    filtering = (corr.iloc[[i], :]) >= 0.77
    select = np.where(filtering == True)[1]
    new_f = df.iloc[:, select]
    df['corr {}_mean'.format(c[i])] = new_f.mean(axis=1)
    df['corr {}_var'.format(c[i])] = new_f.var(axis=1)
    df['corr {}_std'.format(c[i])] = new_f.std(axis=1)
    df['corr {}_sum'.format(c[i])] = new_f.sum(axis=1)
    df['corr {}_max'.format(c[i])] = new_f.max(axis=1)
    df['corr {}_min'.format(c[i])] = new_f.min(axis=1)
    df['corr {}_median'.format(c[i])] = new_f.median(axis=1)
    df['corr {}_maxmin'.format(c[i])] = new_f.max(axis=1) - new_f.min(axis=1)
# 피처를 더 늘려봄 (279개)

model = XGBClassifier(n_estimators=300, n_jobs=-1, learning_rate=0.15, subsample=0.7, max_depth=12, objective="multi:softprob", random_state=42)
```

cross_validation 결과 0.9221962616822431의 조금 더 상승한 정확도가 나옴

실제 제출결과는 **0.925764192139738** 로 역시 acc가 상승한 결과가 나옴 (피처의 수에 비례한다고 생각해 볼 수 있음)