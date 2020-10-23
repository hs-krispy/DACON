##  Grid search optimization

- 지정해준 파라미터들의 모든 조합을 시도해보는 Hyperparameter Optimization 방법

```python
param = {
        'learning_rate': (0.1, 0.45, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'subsample': (0.6, 1.0, 'uniform'),
        'colsample_bytree': (0.6, 1.0, 'uniform'),
        'colsample_bylevel': (0.6, 1.0, 'uniform')
}

xgb = XGBClassifier(n_estimators=400, n_jobs=-1, objective="multi:softprob", tree_method="exact", random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
gcv = GridSearchCV(xgb, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
gcv.fit(x, y)
pp.pprint(gcv.best_params_)   # 최적의 파라미터 값
pp.pprint(gcv.best_score_)    # 최고의 점수

model = gcv.best_estimator_    # 최고의 모델
y_pred = model.predict(Test).tolist()
```



## Bayesian optimization

모든 경우의 수를 시도해보는 Grid search optimization으로 많은 hyperparameter를 시도해보기에는 너무 많은 시간과 컴퓨팅 파워가 소모되어서 Bayesian optimization도 사용해 보았습니다.



- 입력값 x를 받는 미지의 목적 함수(objective function) f를 상정하여, 그 함숫값 f(x)를 최대로 만드는 최적해 x∗를 찾는 것을 목적으로 함
- 목적 함수의 표현식을 명시적으로 알지 못하면서, 하나의 함숫값 f(x)를 계산하는 데 오랜 시간이 소요되는 경우를 가정
- **가능한 한 적은 수의 입력값 후보들에 대해서만 그 함숫값을 순차적으로 조사**하여, f(x)를 최대로 만드는 **최적해 x∗를 *빠르고 효과적으로* 찾는 것**이 주요 목표
- 매 회 새로운 hyperparameter 값에 대한 조사를 수행할 시 **‘사전 지식’을 충분히 반영**하면서, 동시에 전체적인 탐색 과정을 좀 더 체계적으로 수행하기 위한 Hyperparameter Optimization 방법



```python
param = {
        'learning_rate': (0.1, 0.45, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'subsample': (0.6, 1.0, 'uniform'),
        'colsample_bytree': (0.6, 1.0, 'uniform'),
        'colsample_bylevel': (0.6, 1.0, 'uniform')
}

def status_print(optim_result):
    
    pp.pprint('Model #{}\nBest acc: {}\nBest params: {}\n'.format(
        len(all_models),
        bayes_cv_tuner.best_score_,
        bayes_cv_tuner.best_params_
    ))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
xgb = XGBClassifier(n_estimators=400, n_jobs=-1, objective="multi:softprob", tree_method="exact", random_state=42)
bayes_cv_tuner = BayesSearchCV(estimator=xgb, search_spaces=param, scoring='accuracy', cv=skf, n_jobs=-1, n_iter=30, verbose=1, refit=True, random_state=42)
bayes_cv_tuner.fit(x.values, y.values, callback=status_print)
```

