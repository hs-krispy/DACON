## XGBoost

경사 하강법을 이용하는 **Gradient Boosting을 기반**으로 **CART(Classification and  Regression Trees, 다수의 학습방법을 여러가지 Decision Tree(결정 트리)로 정의한 방법 )를 적용**해서 만들어진 모델로 **느리고 overfitting의 가능성이 있는 단점을 보완**함 

### 장점

- Gradient Boosting에 비해 빠름
- overfitting 방지가 가능한 규제가 포함
- CART 기반(분류와 회귀 둘 다 가능)
- 조기 종료(early stopping)를 제공