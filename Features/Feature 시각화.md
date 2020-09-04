## Feature 시각화

피처들을 자세하게 살펴보고 학습에 효과적으로 활용하기 위해 시각화를 진행

#### Feature correlation

피처의 상관관계를 시각화

```python
plt.figure(figsize=(15,15))
sns.heatmap(train[cols].corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/92244856-73cadd00-eefe-11ea-9924-2db755c7d8d7.PNG" width=100% />

#### Feature importance

피처의 중요도를 시각화

```python
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)
# train : test = 0.85 : 0.15
evals = [(test_x, test_y)]
lgbm = LGBMClassifier(n_estimators=2000, learning_rate=0.05, max_depth=12, num_leaves=4000, boosting_type="goss")
lgbm.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
f, ax = plt.subplots(figsize=(15, 15))
plot_importance(lgbm, ax=ax)
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/92245608-94dffd80-eeff-11ea-9f28-c9096348d478.PNG" width=100% /> 

#### 클래스별 피처 분포

```python
scaler = MaxAbsScaler()
train[:][cols] = scaler.fit_transform(train[:][cols]) 
# 'class 피처를 제외하고 정규화'
planet0 = train[y == 0] # 클래스 0
planet1 = train[y == 1] # 클래스 1
planet2 = train[y == 2] # 클래스 2
bin = list(np.arange(0, 1.05, 0.05)) 
# 피처의 값을 0 ~ 1 사이를 0.5 간격으로 나눈 것으로
x_range = np.arange(0, 1.0, 0.05)
for i in cols:
    fig = plt.figure()
    fig.suptitle(i, fontsize=16)
    p0 = fig.add_subplot(3, 1, 1)
    p1 = fig.add_subplot(3, 1, 2)
    p2 = fig.add_subplot(3, 1, 3)
    p0.set_title("class 0")
    total_number = len(planet0) # 클래스 0의 전체 데이터 수
    hist0, bins0 = np.histogram(planet0[i], bin)
    # 각 값들의 빈도와 범위 리턴
    hist0_normal = np.asarray(hist0) / total_number
     # 각 값들의 빈도를 해당 클래스의 전체 데이터 수로 나누어서 정규화 
    p0.set_xlim(0, 1.0) # x축 범위
    p0.set_ylim(0, 1.0) # y축 범위
    p0.bar(x_range, hist0_normal, width=0.2, edgecolor="b")
    p1.set_title("class 1") 
    total_number = len(planet1) # 클래스 1의 전체 데이터 수
    hist1, bins1 = np.histogram(planet1[i], bin) 
    hist1_normal = np.asarray(hist1) / total_number
    p1.set_xlim(0, 1.0)
    p1.set_ylim(0, 1.0)
    p1.bar(x_range, hist1_normal, width=0.2, edgecolor="r")
    p2.set_title("class 2")
    total_number = len(planet2) # 클래스 2의 전체 데이터 수
    hist2, bins2 = np.histogram(planet2[i], bin)
    hist2_normal = np.asarray(hist2) / total_number
    p2.set_xlim(0, 1.0)
    p2.set_ylim(0, 1.0)
    p2.bar(x_range, hist2_normal, width=0.2, edgecolor="k")
    plt.show()
```

 MaxAbsScaler() - 피처의 값들이 -1 ~ 1 사이에(절댓값이 0 ~ 1 사이에) 오도록 재조정(음수 데이터가 존재하기 때문에 선택)