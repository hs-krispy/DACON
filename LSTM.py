import pandas as pd
import numpy as np

np.random.seed(100)
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import he_uniform, GlorotUniform, GlorotNormal
from tensorflow.keras.preprocessing import sequence

train_x_path = "C:/Users/ohs/Downloads/open/train_x_df.csv"
train_y_path = "C:/Users/ohs/Downloads/open/train_y_df.csv"
test_x_path = "C:/Users/ohs/Downloads/open/test_x_df.csv"
sample_submission = "C:/Users/ohs/Downloads/open/sample_submission.csv"

pd.set_option('display.max_rows', 1500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 1500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)


# train_x = pd.read_csv(train_x_path)
# train_y = pd.read_csv(train_y_path)
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
# np.save("Data/test_x.npy", test_x)

train_x = np.load("Data/train_x.npy")
train_y = np.load("Data/train_y.npy")
test_x = np.load("Data/test_x.npy")
print(train_x.shape, train_y.shape, test_x.shape)
data = np.concatenate((train_x, train_y), axis=1)


def plot_series(x_series, y_series):
    # 입력 series와 출력 series를 연속적으로 연결하여 시각적으로 보여주는 코드 입니다.
    plt.plot(x_series, label='input_series')
    plt.plot(np.arange(len(x_series), len(x_series) + len(y_series)),
             y_series, label='output_series')
    plt.axhline(1, c='red')
    plt.legend()


# # sample_id 1012에 해당하는 sample의 분단위 시가 변동 정보 시각화
idx = 1012
plt.figure(figsize=(10, 7))
plot_series(train_x[idx, :, 2], train_y[idx, :, 2])


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


timestep = 120
step = 1
delay = 1
batch = 1
buffer_size = 1000


def process_data(data, timestep):
    train = []
    label = []
    for i in range(len(data) - timestep):
        train.append(data[i: i + timestep])
        label.append(data[i + timestep])

    return np.reshape(train, (-1, timestep, 1)), np.ravel(label)


X_train, y_train = process_data(data[idx, :train_x.shape[1], 1], timestep)
X_val, y_val = process_data(data[idx, data.shape[1] - train_y.shape[1] - timestep:, 1], timestep)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

optimizer = Adam(lr=0.00001)

model = Sequential()
model.add(LSTM(32, activation="tanh", batch_input_shape=(batch, timestep, 1), return_sequences=True, stateful=True,
               kernel_initializer=GlorotUniform))
model.add(LayerNormalization())
model.add(LSTM(64, activation="tanh", batch_input_shape=(batch, timestep, 1), return_sequences=True, stateful=True,
               kernel_initializer=GlorotUniform))
model.add(LayerNormalization())
model.add(LSTM(128, activation="tanh", batch_input_shape=(batch, timestep, 1), return_sequences=True, stateful=True,
               kernel_initializer=GlorotUniform))
model.add(LayerNormalization())
model.add(LSTM(256, activation="tanh", batch_input_shape=(batch, timestep, 1), stateful=True,
               kernel_initializer=GlorotUniform))
model.add(LayerNormalization())
model.add(Dense(1))
model.compile(loss="mae", optimizer=optimizer)
model.summary()

early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
model.fit(X_train, y_train, epochs=5, batch_size=batch, verbose=1, callbacks=[early_stop])
pred = model.predict(X_val, batch_size=batch)
print(pred)
plt.plot(range(1380, 1380 + 120), np.ravel(pred), label="prediction")
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(np.ravel(pred), train_y[idx, :, 1]))
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
        train, label = process_data(test_x[idx, :, 1], timestep)
        data = test_x[idx, -timestep:, 1].tolist()
        pred_y = []
        model = Sequential()
        model.add(
            LSTM(32, activation="tanh", batch_input_shape=(batch, timestep, 1), return_sequences=True, stateful=True,
                 kernel_initializer=GlorotUniform))
        model.add(LayerNormalization())
        model.add(
            LSTM(64, activation="tanh", batch_input_shape=(batch, timestep, 1), return_sequences=True, stateful=True,
                 kernel_initializer=GlorotUniform))
        model.add(LayerNormalization())
        model.add(
            LSTM(128, activation="tanh", batch_input_shape=(batch, timestep, 1), return_sequences=True, stateful=True,
                 kernel_initializer=GlorotUniform))
        model.add(LayerNormalization())
        model.add(LSTM(256, activation="tanh", batch_input_shape=(batch, timestep, 1), stateful=True,
                       kernel_initializer=GlorotUniform))
        model.add(LayerNormalization())
        model.add(Dense(1))
        model.compile(loss="mae", optimizer=optimizer)
        model.fit(train, label, epochs=5, batch_size=batch, verbose=1, callbacks=[early_stop])

        while True:
            if len(pred_y) == 120:
                break
            input = np.reshape(data, (-1, timestep, 1)).astype('float32')
            pred = np.ravel(model.predict(input, batch_size=batch))
            pred_y.extend(pred)
            del data[0]
            data.append(pred)

        test_pred_array[idx, :] = pred_y
    except:
        print(idx, " 샘플은 수렴하지 않습니다.")
        pass

submission = array_to_submission(test_x, test_pred_array)
submission.to_csv("Data/submission_propher.csv", index=False)
# submission - 9676.0121104967
