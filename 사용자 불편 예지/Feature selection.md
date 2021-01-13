## Feature selection

### RFE

<img src="https://user-images.githubusercontent.com/58063806/104395051-3ac90600-558b-11eb-959d-8bdad82a567e.png" width=100%/>

64개의 피처가 중 절반인 32개 부터 63개의 피처까지 selection 

```python
X = train
y = problem

lgb = LGBMClassifier(random_state=42)
res = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for count in range(32, X.shape[1]):
    rfe = RFE(lgb, n_features_to_select=count)
    fit = rfe.fit(X, y)
    fs = X.columns[fit.support_].tolist()
    selected_x = X[fs]
    print(selected_x)
    i = 0
    score = []
    for train_idx, valid_idx in skf.split(selected_x, y):
        train_x, valid_x = selected_x.iloc[train_idx], selected_x.iloc[valid_idx]
        train_y, valid_y = y[train_idx], y[valid_idx]
        evals = [(valid_x, valid_y)]
        lgb.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
        valid_prob = lgb.predict_proba(valid_x)[:, 1]
        auc_score = roc_auc_score(valid_y, valid_prob)
        score.append(auc)
    res.append([count, np.mean(score)])

print(res)
```

```python
# 기존에 모든 피처를 사용했을때 validation score - 0.8147328 
[[32, 0.8111534500000002]
 [33, 0.8105618] 
 [34, 0.8126507999999999] 
 [35, 0.81230515]
 [36, 0.81257635]
 [37, 0.8123688]
 [38, 0.8131268500000001]
 [39, 0.813312]
 [40, 0.8142455999999999]
 [41, 0.8127405]
 [42, 0.8123331500000001]
 [43, 0.8135006]
 [44, 0.81394515]
 [45, 0.8138977000000001]
 [46, 0.81337595]
 [47, 0.8141035499999999]
 [48, 0.8146181]
 [49, 0.8154313999999999] # 49개의 feature를 사용할 때 가장 높은 score가 나타남
 [50, 0.8135337499999998]
 [51, 0.8134140499999999]
 [52, 0.8134612499999999]
 [53, 0.8144387]
 [54, 0.8142391]
 [55, 0.8140034]
 [56, 0.81486615]
 [57, 0.8149497]
 [58, 0.8150414500000001]
 [59, 0.8148565]
 [60, 0.8146147000000001]
 [61, 0.8148611000000001]
 [62, 0.8148611000000001]
 [63, 0.8148611000000001]]

# submission score - 0.8142029657
# validation 보다 낮은 score가 나옴
```

<img src="https://user-images.githubusercontent.com/58063806/104397794-dc068b00-5590-11eb-8d30-1ae927a24892.png" width=100% />

quality를 제외하고 err data만 이용해서 30 ~ 50개의 피처 selection

```python
# 기존에 모든 피처를 사용했을때 validation score - 0.8145382
[[30, 0.81165015]
 [31, 0.8101331]
 [32, 0.81305935]
 [33, 0.8131881]
 [34, 0.8130145000000001]
 [35, 0.8134457999999999]
 [36, 0.8131423999999999]
 [37, 0.8129899]
 [38, 0.81335675]
 [39, 0.8130940499999999]
 [40, 0.8134102999999999]
 [41, 0.8137716000000002]
 [42, 0.8138569499999999]
 [43, 0.81417295]
 [44, 0.8141775499999999]
 [45, 0.8144997]
 [46, 0.8140024499999999]
 [47, 0.81356125]
 [48, 0.8140735000000001]
 [49, 0.81377325]
 [50, 0.8148294500000001]] 
# 기존 데이터에서 하나가 빠진 50개의 피처를 사용할때 가장 높은 score가 나타남
# submission score - 0.8172164724, 모든 피처를 사용한 경우와 거의 차이없음	
```

