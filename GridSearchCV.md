## GridSearchCV

각각의 파라미터로 모델을 여러개 생성해서 최적의 파라미터를 찾아주는 함수

```python
# 파라미터 후보군
param_grid = {
                 'max_depth': list(range(7, 16)),
                 'colsample_bytree': list(np.arange(0.5, 1.05, 0.05)),
                 'colsample_bylevel':list(np.arange(0.5, 1.05, 0.05)),
                 'subsample' : list(np.arange(0.5, 1.05, 0.05)),
                 'learning_rate' : list(np.arange(0.01, 0.11, 0.01)),
                 'n_estimators':[2000],
                 'objective':["multi:softmax"],
                 'random_state':[42]
            }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 교차 검증까지 진행

model = XGBClassifier()
gcv = GridSearchCV(model, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
gcv.fit(x, y)
print('best params', gcv.best_params_)   # 최적의 파라미터 값
print('best score', gcv.best_score_)    # 최고의 점수
```

