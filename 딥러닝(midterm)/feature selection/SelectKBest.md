### SelectKBest를 이용한 feature selection

```python
c = x.columns.tolist()
select = SelectKBest(score_func=chi2, k=x.shape[1])
fit = select.fit(x, y)
f_order = np.argsort(-fit.scores_)
sorted_col = x.columns[f_order].tolist() # 정렬된 피처들
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(data=x, columns=c)
model = XGBClassifier(tree_method='gpu_hist', n_estimators=200, n_jobs=-1, learning_rate=0.25, max_depth=13, objective="multi:softmax", random_state=42)
res = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
for idx in range(1, x.shape[1] + 1):
    sf = sorted_col[0:idx]
    selected_x = x[sf]
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
    res.append([idx, acc.mean(), acc.std()])
    # print(str(model) + " average accuary :", acc.mean(), " std :", acc.std())

print(res)
```

```python
[[1, 0.5939252336448598, 0.01549120042453344], 
 [2, 0.6219626168224299, 0.02062966360247113], 
 [3, 0.6364485981308411, 0.02207171154882158], 
 [4, 0.6448598130841122, 0.01968726582517841], 
 [5, 0.6616822429906544, 0.022307884602392553], 
 [6, 0.6728971962616822, 0.01996262557317182], 
 [7, 0.8434579439252337, 0.027005950050093243], 
 [8, 0.8483644859813083, 0.020646856644967633], 
 [9, 0.8623831775700934, 0.013958202832938188], 
 [10, 0.8689252336448599, 0.015122075818022806], 
 [11, 0.8728971962616823, 0.013266420155680745], 
 [12, 0.8813084112149532, 0.01049062818721665], 
 [13, 0.8785046728971964, 0.013822616315653314], 
 [14, 0.8838785046728971, 0.01645662771302836], 
 [15, 0.9077102803738317, 0.01414468305389281], 
 [16, 0.908411214953271, 0.01506239388905253], 
 [17, 0.907943925233645, 0.01571511375775723], 
 [18, 0.9088785046728972, 0.014628482095793028], 
 [19, 0.9077102803738318, 0.015186915887850455], 
 [20, 0.908177570093458, 0.015251480299408962], 
 [21, 0.9109813084112149, 0.015122075818022811], 
 [22, 0.9109813084112149, 0.013317757009345791], 
 [23, 0.9098130841121496, 0.014448247508168515], 
 [24, 0.910747663551402, 0.014732600720032194], 
 [25, 0.9102803738317757, 0.015853453695034943], 
 [26, 0.9098130841121496, 0.01459859752505916], 
 [27, 0.9088785046728972, 0.01594957772511777], 
 [28, 0.9088785046728972, 0.015568533877530554], 
 [29, 0.9116822429906544, 0.013249950353979278], 
 [30, 0.9100467289719626, 0.01606042883746205], 
 [31, 0.913785046728972, 0.014267649196983217]]
```

selectKBest 방식에서는 변수를 모두 사용하는 것이 가장 높은 acc를 기록