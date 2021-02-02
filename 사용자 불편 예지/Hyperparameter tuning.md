## Hyperparameter tuning

### Bayesian Search

```python
import pandas as pd
import numpy as np
import pprint as pp
from lightgbm import LGBMClassifier, plot_importance
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
# --------------------------- 데이터셋 구성을 위한 변수 ----------------------------
# train 데이터는 user_id가 10000부터 24999까지 총 15000개가 연속적으로 존재.
train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000
# test 데이터는 user_id가 30000부터 44998까지 총 14999개가 존재.
test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

# --------------------------- 생성한 Dataset load, reformatting ----------------------------
train = pd.read_csv("final_train.csv")
test = pd.read_csv("final_test.csv")

# --------------------------- input 데이터에 대한 클래스 생성(불만 제기 여부) ----------------------------
train_prob = pd.read_csv("data/train_problem_data.csv")
problem = np.zeros(15000)
# person_idx의 problem이 한 번이라도 발생했다면 1, 없다면 0
problem[train_prob.user_id.unique() - train_user_id_min] = 1

# --------------------------- 학습에 이용할 데이터 설정, 교차 검증을 통해 모델 구축 ----------------------------
X = train
y = problem
print(X.shape)
print(y.shape)

param = {
                 'max_depth': list(np.arange(7, 15, 2)),
                 'num_leaves': list(np.arange(31, 101, 10)),
                 'colsample_bytree': list(np.arange(0.6, 1.05, 0.1)),
                 'subsample': list(np.arange(0.6, 1.05, 0.1)),
                 'learning_rate': list(np.arange(0.1, 0.4, 0.1))
            }

def status_print(optim_result):
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    pp.pprint('Model #{}\nBest acc: {}\nBest params: {}\n'.format(
        len(all_models),
        bayes_cv_tuner.best_score_,
        bayes_cv_tuner.best_params_
    ))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb = LGBMClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
bayes_cv_tuner = BayesSearchCV(estimator=lgb, search_spaces=param, scoring='roc_auc', cv=skf, n_jobs=-1, n_iter=50,
                               verbose=1, refit=True, random_state=42)
bayes_cv_tuner.fit(X, y, callback=status_print)
best_model = bayes_cv_tuner.best_estimator_
y_pred = best_model.predict_proba(test)[:, 1]
sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission['problem'] = y_pred.reshape(-1)
sample_submission.to_csv("final_submission.csv", index=False)
```



### Grid Search

```python
param_grid = {
                 'max_depth': list(np.arange(7, 15, 2)),
                 'num_leaves': list(np.arange(31, 101, 20)),
                 'colsample_bytree': list(np.arange(0.6, 1.05, 0.1)),
                 'subsample' : list(np.arange(0.6, 1.05, 0.1)),
                 'learning_rate' : list(np.arange(0.1, 0.4, 0.1))
            }

lgb = LGBMClassifier(n_estimators=1000, n_jobs=-1, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gcv = GridSearchCV(lgb, param_grid=param_grid, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
gcv.fit(X, y)
print('best params', gcv.best_params_)   # 최적의 파라미터 값
print('best score', gcv.best_score_)    # 최고의 점수


now = dt.datetime.now()
print(now)

model = gcv.best_estimator_    # 최고의 모델
y_pred = model.predict_proba(test)[:, 1]
sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission['problem'] = y_pred.reshape(-1)
sample_submission.to_csv("submission.csv", index=False)
```

