## LightGBM

level wise (균형 트리) 분할을 사용한 기존의 gradient boosting 알고리즘과 다르게 leaf wise (리프 중심) 트리 분할을 사용

트리의 균형을 맞추지 않고 **max delta loss 값을 가지는 리프 노드를 지속적으로 분할**하면서 진행하는데 비대칭적이고 깊은 트리가 생성되지만 동일한 leaf를 생성할 때 level wise보다 손실을 줄일 수 있음

### 장점

- 속도가 빠름
- 적은 메모리를 사용
- 정확도가 높음
- GPU활용 가능

<img src="https://user-images.githubusercontent.com/58063806/91656909-97091d00-eaf7-11ea-8075-e58236003b8c.JPG" width=60% />