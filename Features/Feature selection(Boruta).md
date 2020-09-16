## Feature selection(Boruta)

**랜덤포레스트(RandomForest)**를 기반으로하는 변수선택기법

1. 모든 피쳐들을 복사해서 새로운 칼럼을 생성(shadow feature)
2. 복사한 피쳐들 (섀도우 피쳐) 각각을 따로 섞음
3. 섀도우 피쳐들에 대해서 랜덤 포레스트를 실행하고, Z score를 얻음
4. 얻은 Z score 중에서 최댓값인 MSZA를 찾음 (MSZA, Max Z-score among shadow attributes)
5. 기존 피쳐들에 대해서 랜덤 포레스트를 실행하여 Z score를 얻고 각각의 기존 피쳐들에 대해서 Z-score > MSZA 인 경우 히트 수를 올린다.
6. Z-score <= MSZA인 경우, MSZA에 대해서 two-side equality test를 수행
7. 통계적으로 유의한 수준에서 Z-score < MSZA인 경우, 해당 피쳐를 중요하지 않은 피쳐로 드랍
8. 통계적으로 유의한 수준에서 Z-score > MSZA인 경우, 해당 피쳐를 유지
9. 모든 피쳐들의 중요성이 결정되거나 최대 반복 회수에 도달할 때까지 Step 5부터 반복

``` python
rf = RandomForestClassifier(criterion="entropy", n_jobs=-1, max_depth=30, n_estimators=250, max_features=11, random_state=42, verbose=True)
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter=10, perc=90)
boruta_feature_selector.fit(x, y)
x_filtered = boruta_feature_selector.transform(x)
final_features = list()
indexes = np.where(boruta_feature_selector.support_ == True)
for i in np.nditer(indexes):
    final_features.append(cols[i])
print(final_features)
```

<img src="https://user-images.githubusercontent.com/58063806/93356257-ebdbc000-f879-11ea-86e0-b6d40d7b7781.PNG" width=100% />

기존의 18개 피처에서 5개의 피처('airmass_u', 'airmass_g', 'airmass_r', 'airmass_i', 'airmass_z')가 제거된 것을 알 수 있음

(실험 결과 반복횟수를 더 늘려봐도 5개만 제거됨)