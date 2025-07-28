from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from dudoachungkhoan import ALL_HAM


def hamDuDoan():
    x_train , y_train ,scaled_data ,scaler , training_data_len , dataset , data = ALL_HAM()
    # Tải lại mô hình
    model = load_model('model/model.h5')
    # Tạo tập dữ liệu thử nghiệm
    # Tạo một mảng mới chứa các giá trị được chia tỷ lệ từ chỉ số 1543 đến 2002
    test_data = scaled_data[training_data_len - 60: , :]
    # Tạo bộ dữ liệu x_test và y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Chuyển đổi dữ liệu thành một mảng có nhiều mảng
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Nhận các mô hình dự đoán giá trị giá
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Lấy lỗi bình phương trung bình gốc (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    return data , predictions , rmse , training_data_len


def HienThi ():
    data , predictions , rmse ,training_data_len = hamDuDoan()

    # Vẽ đồ thị dữ liệu
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Trực quan hóa dữ liệu
    plt.figure(figsize=(16,6))
    plt.title('Mô hình huấn luyện')
    plt.xlabel('Thời gian', fontsize=18)
    plt.ylabel('Giá đóng cửa USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Huấn luyện', 'Giá trị thực', 'Giá dự đoán'], loc='lower right')
    plt.show()

# HienThi()