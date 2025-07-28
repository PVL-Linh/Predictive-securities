from keras.models import Sequential
from keras.layers import Dense, GRU , SimpleRNN , LSTM 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from dudoachungkhoan import ALL_HAM


def train_model_lstm ():

    x_train , y_train ,scaled_data ,scaler , training_data_len , dataset , data = ALL_HAM()
    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Huấn luyện mô hình
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    model.save('model/model.h5')
    print ('đã lưu model ')




def train_model_GRU():
    x_train, y_train, scaled_data, scaler, training_data_len, dataset, data = ALL_HAM()
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(GRU(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    model.save('model/model.h5')
    print('Đã lưu model GRU.')




def train_model_RNN():
    x_train, y_train, scaled_data, scaler, training_data_len, dataset, data = ALL_HAM()
    model = Sequential()
    model.add(SimpleRNN(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    model.save('model/model.h5')
    print('Đã lưu model RNN.')

