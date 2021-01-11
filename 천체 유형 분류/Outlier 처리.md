## Outlier 처리

각 클래스 별로 최대값과 최소값을 비교

```python
planet0 = train[y == 0] # 클래스 0
planet1 = train[y == 1] # 클래스 1
planet2 = train[y == 2] # 클래스 2
for i in cols:
    print(i, ": ", max(planet0[i]), min(planet0[i]))
    print(i, ": ", max(planet1[i]), min(planet1[i]))
    print(i, ": ", max(planet2[i]), min(planet2[i]))
```

<img src="https://user-images.githubusercontent.com/58063806/92321334-151f7380-f064-11ea-909e-90e94c487703.JPG" width=40% />

<img src="https://user-images.githubusercontent.com/58063806/92321336-1650a080-f064-11ea-8d43-de646e85d834.JPG" width=40% />

<img src="https://user-images.githubusercontent.com/58063806/92321337-16e93700-f064-11ea-800f-c5f6bf4ac045.JPG" width=40% />

위의 결과를 보면

- class 0 - redshift

- class 1 - g, i, z, dered_g, dered_i, dered_z
- class 2 - i, z, dered_i, dered_z

등의 feature 값에 눈에 띄는 이상치가 있는 것을 볼 수 있음

### 이상치 처리 함수

```python
def detect_outliers(df, features):
    for i in range(3):
        for col in features:
            Q1 = np.percentile(df[i][col], 25)
            Q2 = np.percentile(df[i][col], 50)
            Q3 = np.percentile(df[i][col], 75)
            IQR = Q3 - Q1
            outlier_step = 2.0 * IQR
            df[i][(df[i][col] < Q1 - outlier_step) | (df[i][col] > Q3 + outlier_step)] = Q2 # 이상치라고 판별된 값들을 중앙값으로 바꿔줌
```

<img src="https://user-images.githubusercontent.com/58063806/92323113-012e3e80-f071-11ea-8434-430110834158.JPG" width=40% />

<img src="https://user-images.githubusercontent.com/58063806/92323114-025f6b80-f071-11ea-94ca-7c65a9cfb554.JPG" width=40% />

<img src="https://user-images.githubusercontent.com/58063806/92323115-02f80200-f071-11ea-87e0-9d4c11d7a52c.JPG" width=40% />