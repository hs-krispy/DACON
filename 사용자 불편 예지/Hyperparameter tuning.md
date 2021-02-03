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
                 'max_depth': list(np.arange(13, 16, 1)),
                #  'colsample_bytree' : list(np.arange(0.6, 1.00, 0.05)),
                #  'num_leaves' : list(np.arange(31, 100, 10)),
                #  'reg_alpha': list(np.arange(0.01, 0.5, 0.05)),
                #  'reg_lambda': list(np.arange(0.01, 0.5, 0.05))
                 'learning_rate': list(np.arange(0.001, 0.01, 0.001))
            }
def status_print(optim_result):
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    pp.pprint('Model #{}\nBest auc: {}\nBest params: {}\n'.format(
        len(all_models),
        bayes_cv_tuner.best_score_,
        bayes_cv_tuner.best_params_
    ))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb = LGBMClassifier(n_estimators=5000, colsample_bytree=0.6, num_leaves=50,
                     min_child_samples=15, n_jobs=-1, reg_alpha=0.46, reg_lambda=0.11, random_state=42)
bayes_cv_tuner = BayesSearchCV(estimator=lgb, search_spaces=param, scoring='roc_auc', cv=skf, n_jobs=-1, n_iter=200,
                               verbose=1, refit=True, random_state=42)
bayes_cv_tuner.fit(X, y, callback=status_print)
best_model = bayes_cv_tuner.best_estimator_
y_pred = best_model.predict_proba(test)[:, 1]
sample_submission = pd.read_csv('Data/sample_submission.csv')
sample_submission['problem'] = y_pred.reshape(-1)
sample_submission.to_csv("final_submission.csv", index=False)
```



Grid Search에 비해 비교적 시간이 적게들고 좋은 결과를 낼 수 있는 Bayesian Search를 이용해 parameter들의 범위를 점점 줄여나가면서 튜닝을 진행   

```python
lgb = LGBMClassifier(n_estimators=1500, max_depth=13, learning_rate=0.01, colsample_bytree=0.8, min_child_samples=15, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.1, random_state=42)
# validation - 0.8213191999999999
# submission - 0.8209303742	

lgb = LGBMClassifier(n_estimators=1500, max_depth=13, learning_rate=0.01, colsample_bytree=0.6, min_child_samples=15, n_jobs=-1, reg_alpha=0.26, reg_lambda=0.11, random_state=42)
# validation - 0.8216709

lgb = LGBMClassifier(n_estimators=1500, max_depth=13, learning_rate=0.01, colsample_bytree=0.6, num_leaves=50, min_child_samples=15, n_jobs=-1, reg_alpha=0.26, reg_lambda=0.11, random_state=42)
# validation - 0.8221755
# submission - 0.8229939643	

lgb = LGBMClassifier(n_estimators=1500, max_depth=13, learning_rate=0.01, colsample_bytree=0.6, num_leaves=50, min_child_samples=15, n_jobs=-1, reg_alpha=0.46, reg_lambda=0.11, random_state=42)
# validation - 0.8222208

lgb = LGBMClassifier(n_estimators=2000, max_depth=13, learning_rate=0.005, colsample_bytree=0.6, num_leaves=50, min_child_samples=15, n_jobs=-1, reg_alpha=0.46, reg_lambda=0.11, random_state=42)
# validation - 0.8224834000000001
# Final submission - 0.8229607068	
```

public score - 0.82299 (전체 테스트 data의 33%)

private score - 0.82068 (전체 테스트 data의 67%)

