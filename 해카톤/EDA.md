## EDA

#### 고객 정보

- 차량 구매 이력이 있는 고객의 정보
- 고객별로 주소 data는 없을 수 있음

우선적으로 결측값 확인

```python
print(customer.isnull().sum())

# CUS_ID                    0	고객 ID
# PSN_BIZR_YN          957264	개인사업자여부
# SEX_SCN_NM                0	성별
# TYMD                      0	생년월일
# CUS_ADM_TRY_NM       144396	주소_행정시도명
# CUS_N_ADMZ_NM        147294	주소_시군구명
# CUS_ADMB_NM          855306	주소_행정동명
# CLB_HOUS_PYG_NM      473289	주택 평형
# REAI_BZTC_AVG_PCE    508626	주택 평균가격
```

- PSN_BIZR_YN는 개인사업자여부로 **Y인 경우에만 사업자등록이 된 고객**이고 null값인 고객은 그렇지 않은 경우로 판단
- 나머지 결측값이 있는 컬럼들은 고객의 주소와 주택 정보에 해당
- 주소나 주택의 정보가 있는 고객들은 이러한 정보도 활용 가능

```python
import qgrid

qgrid_widget = qgrid.show_grid(customer[customer["TYMD"] > 20210100], show_toolbar=True)
qgrid_widget
```

<img src="https://user-images.githubusercontent.com/58063806/115995096-d9e2ac80-a614-11eb-8295-b84c04e80515.png" width=100% />

다음과 같이 생년월일이 제대로 기재되어 있지 않은 고객 23명 확인

```python
# 개인 사업자
PSN_Y = customer[customer["PSN_BIZR_YN"] == "Y"]["CUS_ID"].tolist()
print(len(PSN_Y))
# 138942

df = cars[cars["CUS_ID"].isin(PSN_Y)].groupby("CUS_ID")["CAR_ID"].count() >= 2
select_cid = df[df == True].index.tolist()
print(len(select_cid))
# 95150
```

**개인 사업자로 등록된 고객은 138942**명이고 그 중, 차량 출고정보가 2개 이상있는 고객은 95150명으로 70% 이상 해당

고객들의 주택 평균가격을 확인

```python
print(customer["REAI_BZTC_AVG_PCE"].mean(axis=0))
# 47164.57554715953
```

전체적으로 가격들을 살펴보았을때 단위는 천원 단위로 가정 (평균적인 가격은 4억 7천만원 정도로 확인)

```python
selected_cus = customer[customer["REAI_BZTC_AVG_PCE"] >= customer["REAI_BZTC_AVG_PCE"].mean(axis=0)]["CUS_ID"].unique()
```

주택가격이 평균 이상인 고객들을 추출 (200105 명)

```python
select = df.groupby("CUS_ID")["CAR_ID"].count() >= 2
selected_cus2 = select[select == True].index
new_cus = list(set(selected_cus) & set(selected_cus2))
print(len(new_cus))
# 119481
```

주택가격이 평균 이상이면서 차량구매이력이 2개 이상인 고객들을 추출(119481 명)

```python
df.groupby("CUS_ID")["WHOT_DT"].unique()
df.groupby("CUS_ID")["CAR_HLDG_FNH_DT"].unique()
```

해당 고객들의 차량 출고일과 차량 보유종료일

```python
area = customer[customer["CUS_ID"].isin(new_cus)]["CUS_N_ADMZ_NM"].value_counts()
```

해당 고객들의 주소_시군구명 파악

<img src="https://user-images.githubusercontent.com/58063806/115731121-d5728580-a3c1-11eb-9a8a-aab652263b43.png" width=60% />

위의 주소정보를 바탕으로 주택에 대한 정보가 없는 고객들의 주택 정보를 대략적으로 파악가능



#### 차량 정보

- 차량의 정보와 고객의 차량 출고, 보유에 관한 일자와 구매이력이 있으며 대차, 추가구매 추정에 있어 중요한 정보가 될 것으로 판단

우선적으로 결측값 확인 

```python
print(cars.isnull().sum())

# CAR_ID                   0	차량 ID
# CUS_ID                   0	고객 ID
# WHOT_DT                  0	출고일자
# CAR_HLDG_STRT_DT         0	보유시작일자
# CAR_HLDG_FNH_DT     894153	보유종료일자
# CAR_NM                   0	차량명
# CAR_CGRD_NM_1            0	차량등급명1
# CAR_CGRD_NM_2            0	차량등급명2
# CAR_ENG_NM               0	엔진타입명
# CAR_TRIM_NM            287	트림명
```

전체 데이터의 갯수 1835830개의 절반에 미칠 정도로 보유종료일자에 결측값(NULL)이 상당히 많은데 이는 차량을 처분하지 않은 상태 

CAR_TRIM_NM(트림)은 자동차를 구매할 때 **부가적인 옵션**을 지칭한다고 함

```python
trim_null = cars[cars["CAR_TRIM_NM"].isnull()]

qgrid_widget = qgrid.show_grid(trim_null, show_toolbar=True)
qgrid_widget
```

<img src="https://user-images.githubusercontent.com/58063806/115402132-19308800-a226-11eb-80fc-5a2e8ccefa54.png" width=100% />

**2008/06/17 ~ 2009/08/27 기간에 출고된 차량등급이 RV 중형 SUV인 2.2 Type의 엔진을 가진 일부 싼타페**의 경우에서 모든 트림명 결측치가 발생

해당 시기의 싼타페 차량의 경우 CLX, MLX, SLX 옵션이 있는데 CLX << MLX << SLX 순서로 더 좋은 옵션이라고 함

**CAR_CGRD_NM_1(차량등급명1)**

승용, RV(레저용 자동차), 해당없음 (3가지의 항목이 존재)

차량등급명1이 **해당없음에 해당하는 데이터들을 보면 2007/01/03 ~ 2007/12/04 기간에 출고된 베르나, 쏘나타, 클릭 3가지 차 종류**가 해당되며 **CAR_ENG_NM(엔진타입명)과 CAR_TRIM_NM(트림명) 또한 해당없음**으로 나타남 

비슷한 시기에 동일한 차량에 대한 정보를 보면 CAR_CGRD_NM_1이 대부분 승용으로 나타남 **(CAR_CGRD_NM_1이 해당없음인 데이터들도 승용으로 봐도 무방)**

```python
select = cars.groupby("CUS_ID")["CAR_ID"].count() >= 2
selected_cus = select[select == True].index
select_cars = cars[cars["CUS_ID"].isin(selected_cus)]
qgrid_widget = qgrid.show_grid(select_cars, show_toolbar=True)
qgrid_widget
```

<img src="https://user-images.githubusercontent.com/58063806/115413419-1cc90c80-a230-11eb-890f-e6cf7c607cb1.png" width=100% />

- **차량 출고 정보가 2개 이상 존재하는 경우에 대해 조회 (1345394개의 데이터로 전체의 약 70% 정도에 해당)**
- 전체 **1096206명의 고객 중 605770명 (약 55% 정도)의 고객이 차량 출고 정보가 2개 이상 존재** 

고객이 **기존의 차량을 보유종료하는 일자와 다음 차를 출고하는 일자**를 중심으로 대차와 추가구매를 추정

<img src="https://user-images.githubusercontent.com/58063806/115569607-5cf3c200-a2f8-11eb-81c5-8f6feac6fb92.png" width=100% />

**대차로 추정(기존 차량의 보유종료일자와 다음 차량의 출고일자에 거의 차이가 없음)되는 경우**와 **추가구매로 추정(기존 차량의 보유종류일자가 다음 차량의 출고일자보다 훨씬 나중)되는 경우 모두 존재**하는 경우의 고객 정보도 확인

> 이러한 경우에는 고객의 관점에서 추가적으로 어떤 정보를 이용해서 대차와 추가구매를 추정할지 고려해야 함

총 46대의 차종에 대해 출고 빈도를 시각화

```python
import matplotlib.pyplot as plt

car_name = cars["CAR_NM"].value_counts().index.tolist()
counts = cars["CAR_NM"].value_counts().values.tolist()
plt.figure(figsize=(17, 10))
plt.bar(range(len(car_name)), counts)
plt.rc("font", family="Malgun Gothic")
plt.xticks(range(len(car_name)), car_name, rotation=90)
plt.show()
```

**전체 고객의 차량 출고 빈도**

<img src="https://user-images.githubusercontent.com/58063806/116095777-af135980-a6e3-11eb-84d0-de5baf8f2122.png" width=100% />

**개인사업자 등록 고객의 차량 출고 빈도**

<img src="https://user-images.githubusercontent.com/58063806/116096326-2ba63800-a6e4-11eb-9090-8ebcf68f3b7f.png" width=100% />

**개인사업자 미등록 고객의 차량 출고 빈도**

<img src="https://user-images.githubusercontent.com/58063806/116096682-72942d80-a6e4-11eb-934e-605fb0f44b66.png" width=100% />

차종에 있어서는 크게 차이나는 부분없이 비슷한 분포를 보임



#### 접촉 정보

- **고객이 자발적으로 현대자동차에 접촉**한 데이터로 접촉업무명이 중보한 정보가 될 것으로 판단

우선적으로 결측값 확인 (결측치는 없는 것으로 확인됨)

```python
print(contact.isnull().sum())

# CNTC_SN             0		접촉일련번호
# CUS_ID              0		고객 ID
# CNTC_DT             0		접촉일자
# CNTC_CHAN_NM        0		접촉채널명
# CNTC_AFFR_SCN_NM    0		접촉업무명
```

**접촉채널명**

당첨, 방문(대면), 응모, 이벤트, 인터넷, 전화, 참석, 캠페인 (8개의 채널이 존재)

**접촉업무명**

견적, 고객초청행사, 불만상담, 비포서비스, 비포서비스 & 무상점검, 시승센터시승, 

영업활동(TM), 영업활동(대면), 이벤트, 일반상담, 정비 (11개의 업무가 존재)

> 고객이 먼저 접촉한 데이터이므로 영업이 용무일 경우에 구매로 이어질 확률이 높으며 문의 전 검색이나 해당 제품/서비스를 경험해 본 고객일 확률이 높다고 함 
>
> 비포서비스의 경우 차량의 안전한 운행을 위해 기본 성능 점검 및 정비 상담을 제공해주는 것이라고 함 (사전점검)

**당첨, 응모** -  **고객초청행사, 이벤트 (2)**

```python
print(contact[contact.CNTC_CHAN_NM == "당첨"]["CNTC_AFFR_SCN_NM"].unique())
# ['고객초청행사' '이벤트']
print(contact[contact.CNTC_CHAN_NM == "응모"]["CNTC_AFFR_SCN_NM"].unique())
# ['고객초청행사' '이벤트']
```

나머지도 동일한 방식으로 확인

**이벤트** -  **이벤트 (1)**

**캠페인** -  **영업활동(TM), 영업활동(대면) (2)**

**방문(대면)** -  **정비, 비포서비스, 비포서비스&무상점검, 시승센터시승 (4)**

**인터넷** - **견적, 이벤트 (2)**

**전화 - 일반상담, 불만상담 (2)**

**참석 - 고객초청행사 (1)**

- 접촉채널이 당첨, 응모, 이벤트, 참석인 경우에는 오직 고객초청행사, 이벤트 같은 행사 용무로 접촉 

- 접촉채널이 캠페인인 경우에는 오직 영업활동을 용무로 접촉 
- 접촉채널이 방문인 경우 정비, 비포서비스, 시승을 용무로 접촉
- 접촉채널이 전화인 경우 상담을 용무로 접촉
- 접촉채널이 인터넷일 경우 견적과 이벤트를 용무로 접촉

**견적, 영업활동, 시승 등의 용무가 잦은 고객일 수록 추가 구매의 확률이 높다고 판단이 가능**

