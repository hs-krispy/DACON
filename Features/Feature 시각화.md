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
scaler = StandardScaler()
train = scaler.fit_transform(x)
train = train[np.where(y == 1)] # class 1
len_of_data = train.shape[0]

plot_x = np.arange(-0.9, 1.0, 0.2)

fig = plt.figure(figsize=(25, 13))

for index in range(1, 20):
    if index == 19:
        break

    hist, bins = np.histogram(train[:, index - 1], bins=np.arange(-1.0, 1.2, 0.2))
    hist = hist / len_of_data


    ax = fig.add_subplot(5, 4, index)

    ax.set_ylim(0, 1.1)
    ax.title.set_text(cols[index - 1])
    ax.set_xticks(np.arange(-1.0, 1.2, 0.2))
    ax.bar(plot_x, hist, width=0.2, edgecolor='k', alpha=0.7)
    ax.grid(b=True, axis='y', color='gray', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
```

StandardScaler - 각 feature의 평균이 0과 표준편차가 1이 되도록 변환