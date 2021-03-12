import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm

train_x_path = "C:/Users/0864h/Desktop/open/open/train_x_df.csv"
train_y_path = "C:/Users/0864h/Desktop/open/open/train_y_df.csv"
test_x_path = "C:/Users/0864h/Desktop/open/open/test_x_df.csv"
sample_submission = "C:/Users/0864h/Desktop/open/open/sample_submission.csv"

pd.set_option('display.max_rows', 1500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 1500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)

# train_x_df = pd.read_csv(train_x_path)
# train_y_df = pd.read_csv(train_y_path)
# test_x = pd.read_csv(test_x_path)


def df2d_to_array3d(df_2d):
    # 입력 받은 2차원 데이터 프레임을 3차원 numpy array로 변경하는 함수
    feature_size = df_2d.iloc[:, 2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    sample_index = df_2d.sample_id.value_counts().index
    array_3d = df_2d.iloc[:, 2:].values.reshape([sample_size, time_size, feature_size])
    return array_3d


# train_x = df2d_to_array3d(train_x)
# np.save("Data/train_x.npy", train_x)
# train_y = df2d_to_array3d(train_y)
# np.save("Data/train_y.npy", train_y)
# test_x = df2d_to_array3d(test_x)

train_x = np.load("Data/train_x.npy")
train_y = np.load("Data/train_y.npy")
test_x = np.load("Data/test_x.npy")
print(train_x.shape, train_y.shape, test_x.shape)


def plot_series(x_series, y_series):
    # 입력 series와 출력 series를 연속적으로 연결하여 시각적으로 보여주는 코드 입니다.
    plt.plot(x_series, label='input_series')
    plt.plot(np.arange(len(x_series), len(x_series) + len(y_series)),
             y_series, label='output_series')
    plt.axhline(1, c='red')
    plt.legend()


def generator(data, timestep, delay, min_index, max_index, batch_size=32, step=1):
    if max_index is None:
        max_index = len(data) - timestep - 1
    i = min_index + timestep
    while True:
        if i + batch_size >= max_index:
            i = min_index + timestep
        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = np.zeros((len(rows), timestep // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - timestep, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets


idx = 1012
np.random.seed(100)
timestep = 460
step = 1
delay = 120
batch = 16
buffer_size = 1000


def process_data(data, timestep, start, end):
    train = []
    label = []
    test = []
    for i in range(data.shape[0] - timestep):
        if i + timestep + delay >= data.shape[0]:
            test.append(data[i: i + timestep])
        else:
            train.append(data[i: i + timestep])
            label.append(data[i + timestep: i + timestep + delay])
    return np.reshape(train, (-1, timestep, 1)), np.array(label), np.reshape(test, (-1, timestep, 1))


def make_data(data):
    dataset = []
    for count, id in enumerate(data):
        date = datetime.datetime(2021, 1, 31, 0) + datetime.timedelta(minutes=count + 1)
        dataset.append([date, id])

    dataset = pd.DataFrame(dataset, columns=["ds", "y"])
    return dataset


X_train = make_data(train_x[idx, :, 1])
print(X_train.shape)
prophet = Prophet(seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.5)
prophet.fit(X_train)
future_data = prophet.make_future_dataframe(periods=120, freq="min")
forecast_data = prophet.predict(future_data)
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

prophet.plot(forecast_data)

pred_y = forecast_data.yhat.values[-120:]  # 마지막 5일의 예측 데이터
pred_y_lower = forecast_data.yhat_lower.values[-120:]
pred_y_upper = forecast_data.yhat_upper.values[-120:]
# # sample_id 1012에 해당하는 sample의 분단위 시가 변동 정보 시각화

plt.figure(figsize=(10, 7))
plot_series(train_x[idx, :, 1], train_y[idx, :, 1])
plt.rc('font', family="Malgun Gothic")
plt.plot(range(1380, 1380 + 120), pred_y, c="gold", label="predict cost")
plt.plot(range(1380, 1380 + 120), pred_y_lower, c="red", label="predict lower cost")
plt.plot(range(1380, 1380 + 120), pred_y_upper, c="blue", label="predict upp cost")
plt.legend()
plt.show()

rmse = sqrt(mean_squared_error(pred_y, train_y[idx, :, 1]))
print(rmse)

def array_to_submission(x_array, pred_array):
    # 입력 x_arrry와 출력 pred_arry를 통해서
    # buy_quantitiy와 sell_time을 결정
    submission = pd.DataFrame(np.zeros([pred_array.shape[0], 2], np.int64),
                              columns=['buy_quantity', 'sell_time'])
    submission = submission.reset_index()
    submission.loc[:, 'buy_quantity'] = 0.1

    buy_price = []
    for idx, sell_time in enumerate(np.argmax(pred_array, axis=1)):
        print(pred_array[idx, sell_time])
        buy_price.append(pred_array[idx, sell_time])
    buy_price = np.array(buy_price)
    # 115% 이상 상승한하고 예측한 sample에 대해서만 100% 매수
    submission.loc[:, 'buy_quantity'] = (buy_price > 1.15) * 1
    # 모델이 예측값 중 최대 값에 해당하는 시간에 매도
    submission['sell_time'] = np.argmax(pred_array, axis=1)
    submission.columns = ['sample_id', 'buy_quantity', 'sell_time']
    return submission


test_pred_array = np.zeros([test_x.shape[0], 120])
for idx in tqdm(range(test_x.shape[0])):
    try:
        x_series = test_x[idx,:,1]

        x_df = make_data(x_series)

        prophet = Prophet(seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.5)
        prophet.fit(x_df)

        # 120분 테스트 데이터를 예측합니다.
        future_data = prophet.make_future_dataframe(periods=120, freq='min')
        forecast_data = prophet.predict(future_data)

        pred_y = forecast_data.yhat.values[-120:]
        pred_y_lower = forecast_data.yhat_lower.values[-120:]
        pred_y_upper = forecast_data.yhat_upper.values[-120:]

        test_pred_array[idx,:] = pred_y
    except:
        print(idx, " 샘플은 수렴하지 않습니다.")
        pass

submission = array_to_submission(test_x, test_pred_array)
submission.to_csv("Data/submission_propher.csv", index = False)
# submission - 8223.6283312856