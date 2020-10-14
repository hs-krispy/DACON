### RFE를 이용한 feature selection

```python
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(data=x, columns=c)
model = XGBClassifier(tree_method='gpu_hist', n_estimators=200, n_jobs=-1, learning_rate=0.25, max_depth=13, objective="multi:softmax", random_state=42)
res = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
for count in range(15, x.shape[1]):
    rfe = RFE(model, n_features_to_select=count)
    fit = rfe.fit(x, y)
    fs = x.columns[fit.support_].tolist()
    selected_x = x[fs]
    print(selected_x)
    i = 0
    acc = np.zeros(10)
    for train_idx, test_idx in skf.split(selected_x, y):
        train_x, test_x = selected_x.iloc[train_idx], selected_x.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
        evals = [(test_x, test_y)]
        model.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
        accuracy = model.score(test_x, test_y)
        acc[i] = accuracy
        i += 1
    res.append([count, acc.mean(), acc.std()])

print(res)
```

selectKBest에서 acc 90이 넘었던 15개의 피처부터 진행해봄

```python
[[15, 0.9021028037383176, 0.012858960659642076], 
 [16, 0.9039719626168224, 0.010981308411214944], 
 [17, 0.907943925233645, 0.013142393561145068],
 [18, 0.9067757009345796, 0.012294693567217635], 
 [19, 0.9095794392523364, 0.012142791084286246], 
 [20, 0.9123831775700936, 0.014675054904151606], 
 [21, 0.9126168224299066, 0.012416196500548059], 
 [22, 0.9116822429906544, 0.011858483682663104], 
 [23, 0.9154205607476635, 0.011292566330462215], 
 [24, 0.9147196261682243, 0.01467505490415159], 
 [25, 0.9133177570093458, 0.013681697170053601], 
 [26, 0.9130841121495328, 0.013494886988966028], 
 [27, 0.9130841121495328, 0.016448332623175733], 
 [28, 0.9123831775700936, 0.014450136538379661], 
 [29, 0.913551401869159, 0.013862053246907794], 
 [30, 0.9126168224299066, 0.016919501860029593]]
```

backward 방식으로 23개의 피처를 선별했을때 가장 높은 acc가 나옴

이때 선택된 피처

```python
['c3', 'c4', 'c5', 'c8', 'c10', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c31']
```

