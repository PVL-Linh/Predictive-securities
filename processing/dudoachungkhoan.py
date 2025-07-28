from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# AAPL = pd.read_csv('DATA/APPLE_stock_data.csv')
# FPT = pd.read_csv('DATA/FPT_stock_data.csv')
# GOOGLE = pd.read_csv('DATA/GOOGLE_stock_data.csv')
# VIETJET = pd.read_csv('DATA/VIETJET_stock_data.csv')
# Gọi hàm và truyền vào đường dẫn đến các file CSV
#merge = [AAPL, FPT , GOOGLE , VIETJET]
#df = pd.concat([AAPL, FPT , GOOGLE , VIETJET],axis=0)


def get_Data (NAME):
    

    # Loại bỏ cột 'company_name' nếu có
    if 'company_name' in NAME.columns:
        NAME.drop(columns=['company_name'], inplace=True)
    

    # Đảm bảo cột 'Date' là kiểu datetime
    if 'Date' in NAME.columns:
        NAME['Date'] = pd.to_datetime(NAME['Date'])
    else:
        print("Cột 'Date' không có trong DataFrame.")


    # Đặt cột 'Date' làm chỉ mục nếu có
    if 'Date' in NAME.columns:
        NAME.set_index('Date', inplace=True)


    # Định nghĩa ngày bắt đầu và ngày kết thúc
    start_date = '2017-07-01'
    end_date = datetime.now().strftime('%Y-%m-%d')


    # Lọc dữ liệu theo khoảng thời gian mong muốn nếu cột 'Date' đã được đặt làm chỉ mục
    if 'Date' in NAME.index.names:
        filtered_NAME = NAME.loc[start_date:end_date]
    else:
        print("Cột 'Date' không phải là chỉ mục, không thể lọc dữ liệu.")


    return filtered_NAME


def so_Hang (df):
    # Tạo một khung dữ liệu mới chỉ với cột 'Đóng
    data = df.filter(['Close'])
    # Chuyển đổi khung dữ liệu thành một mảng có nhiều mảng
    dataset = data.values
    # Lấy số hàng để huấn luyện mô hình
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    return training_data_len , dataset ,data


def Chia_Data(training_data_len , dataset):

    # Chia tỷ lệ dữ liệu
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Tạo tập dữ liệu huấn luyện
    # Tạo tập dữ liệu đào tạo được chia tỷ lệ
    train_data = scaled_data[0:int(training_data_len), :]
    # Tách dữ liệu thành tập dữ liệu x_train và y_train
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()

    # Chuyển đổi x_train và y_train thành mảng có nhiều mảng
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Định hình lại dữ liệu
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train , y_train , scaled_data  ,scaler
def ALL_HAM():
    GOOGLE = pd.read_csv('data1/data.csv')
    NAME = GOOGLE
    filtered_NAME = get_Data(NAME)
    training_data_len , dataset , data = so_Hang(filtered_NAME)
    x_train , y_train ,scaled_data ,scaler  = Chia_Data(training_data_len , dataset)
    return x_train , y_train ,scaled_data ,scaler , training_data_len , dataset , data 
