# 📈 Dự đoán Giá Cổ Phiếu bằng XGBoost, LSTM và SVR

![stock](https://user-images.githubusercontent.com/yourusername/yourrepo/stock-prediction-banner.png)

## 📌 Mô tả Dự án

Dự án này tập trung vào việc dự đoán giá cổ phiếu dựa trên dữ liệu lịch sử bằng cách kết hợp nhiều mô hình học máy và học sâu:

* **XGBoost**: Mô hình cây tăng cường, phù hợp với dữ liệu có đặc trưng rõ ràng.
* **LSTM (Long Short-Term Memory)**: Mạng nơ-ron hồi tiếp chuyên xử lý chuỗi thời gian.
* **SVR (Support Vector Regression)**: Mô hình hồi quy tuyến tính và phi tuyến hiệu quả với tập dữ liệu nhỏ.

---

## 🧠 Các Bước Thực Hiện

### 1️⃣ Thu thập & Tiền xử lý dữ liệu

* Dữ liệu được lấy từ Yahoo Finance (sử dụng `yfinance` hoặc `pandas_datareader`).
* Tính toán các đặc trưng như: giá đóng cửa, MA (Moving Average), RSI, phần trăm thay đổi...
* Chuẩn hóa dữ liệu với MinMaxScaler hoặc StandardScaler.

### 2️⃣ Huấn luyện các mô hình

#### 🔹 XGBoost:

* Huấn luyện với các đặc trưng kỹ thuật.
* Tối ưu thông qua `GridSearchCV` hoặc `RandomSearchCV`.

#### 🔹 LSTM:

* Tạo chuỗi thời gian từ dữ liệu.
* Xây dựng mạng LSTM nhiều lớp bằng Keras hoặc PyTorch.
* Sử dụng `TimeSeriesGenerator` để huấn luyện theo batch.

#### 🔹 SVR:

* Dùng `sklearn.svm.SVR` với kernel `rbf`, `linear` hoặc `poly`.
* Phù hợp cho tập dữ liệu đã rút gọn đặc trưng.

### 3️⃣ Đánh giá mô hình

* Sử dụng RMSE, MAE và R² để đo hiệu suất.
* Vẽ biểu đồ so sánh giá thực tế và giá dự đoán.

---

## 🛠️ Hướng dẫn Triển khai

```bash
# Cài đặt thư viện cần thiết
pip install yfinance scikit-learn xgboost keras pandas matplotlib
```

```python
# Tải dữ liệu từ Yahoo Finance
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
```

```python
# Huấn luyện XGBoost
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
```

```python
# Huấn luyện LSTM
model = Sequential([
  LSTM(50, return_sequences=True),
  LSTM(50),
  Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## 📊 Kết Quả Mô Hình

* **XGBoost**: Nhanh, mạnh, nhưng khó dự đoán xu hướng dài hạn.
* **LSTM**: Bắt tốt xu hướng, phù hợp chuỗi thời gian phức tạp.
* **SVR**: Hiệu quả với chuỗi ngắn, nhưng cần chuẩn hóa tốt.

<p align="center">
  <img src="https://user-images.githubusercontent.com/yourusername/yourrepo/prediction-results.png" width="600" />
</p>

---

## 🤝 Đóng góp

Chúng tôi chào đón mọi đóng góp:

* Nâng cao mô hình
* Kết hợp thêm chỉ báo kỹ thuật (MACD, Bollinger Bands,...)
* Triển khai web demo dự đoán theo thời gian thực

---

## 📫 Liên hệ

🌐 GitHub: [github.com/PVL-Linh](https://github.com/PVL-Linh)

---

⭐ Nếu bạn thấy dự án hữu ích, hãy **Star** và **Fork** để ủng hộ nhóm phát triển!
