import numpy as np
from dudoachungkhoan import ALL_HAM
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt


def data_processing ():

    x_train , y_train ,scaled_data ,scaler , training_data_len , dataset , data = ALL_HAM()
    model = load_model('model/model.h5')
    # Giả sử bạn đã có mô hình LSTM đã huấn luyện và có biến x_train_final, y_train_final từ trước
    # Tạo dữ liệu đầu vào cho dự đoán
    last_60_days = scaled_data[-365:]  # Lấy dữ liệu 60 ngày cuối cùng để dự đoán

    # Tạo dự đoán cho 1 tháng tới (khoảng 20 ngày làm việc)
    predictions = []
    for i in range(20):
        x_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        pred = model.predict(x_test)
        predictions.append(pred[0, 0])
        last_60_days = np.append(last_60_days[1:], pred[0, 0])

    # Chuyển đổi dự đoán từ scaled về giá thực
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))



    # Tạo ngày cho dự đoán
    last_date = data.index[-1]
    forecast_dates = [last_date + pd.DateOffset(days=x) for x in range(1, 21)]  # Dự đoán cho 20 ngày tới

    # Tạo DataFrame cho dự đoán
    predictions_df = pd.DataFrame(data=predictions, index=forecast_dates, columns=['Predictions'])

    # Kết hợp dữ liệu thực tế và dự đoán
    combined_df = pd.concat([data, predictions_df], axis=1)

    # Cắt dữ liệu huấn luyện và dự đoán
    train = combined_df[:training_data_len]
    valid = combined_df[training_data_len:]

    return train , valid

    

def truc_quan_bieu_do_mien ():
    train , valid = data_processing()
    # Trực quan hóa dữ liệu
    plt.figure(figsize=(16,6))
    plt.title('Dự đoán Giá Đóng Cửa')
    plt.xlabel('Thời gian', fontsize=18)
    plt.ylabel('Giá đóng cửa USD ($)', fontsize=18)

    # Vẽ dữ liệu huấn luyện và thực tế
    plt.plot(train['Close'], label='Dữ liệu Huấn Luyện')
    plt.plot(valid['Close'], label='Dữ liệu Thực Tế')
    plt.plot(valid['Predictions'], label='Dự Đoán', linestyle='-')

    plt.legend(loc='lower right')
    plt.show()

def truc_quan_bieu_do_cot ():
    train , valid = data_processing()
    # Trực quan hóa dữ liệu bằng cột
    plt.figure(figsize=(16,6))
    plt.title('Dự đoán Giá Đóng Cửa (Cột)')
    plt.xlabel('Thời gian', fontsize=18)
    plt.ylabel('Giá đóng cửa USD ($)', fontsize=18)

    # Vẽ cột cho dữ liệu thực tế và dự đoán
    plt.bar(train.index, train['Close'], color='blue', label='Dữ liệu Huấn Luyện')
    plt.bar(valid.index, valid['Close'], color='green', width=1, label='Dữ liệu Thực Tế')
    plt.bar(valid.index, valid['Predictions'], color='red', width=0.5, alpha=0.5, label='Dự Đoán')

    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.show()

# truc_quan_bieu_do_mien()
# truc_quan_bieu_do_cot()