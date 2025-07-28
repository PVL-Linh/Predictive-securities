# 📈 Stock Price Prediction using XGBoost, LSTM, and SVR

![stock](https://user-images.githubusercontent.com/yourusername/yourrepo/stock-prediction-banner.png)

## 📌 Project Description

This project focuses on predicting stock prices based on historical data using a combination of machine learning and deep learning models:

* **XGBoost**: A powerful boosting tree-based model suitable for structured data.
* **LSTM (Long Short-Term Memory)**: A type of recurrent neural network tailored for time series.
* **SVR (Support Vector Regression)**: A linear/non-linear regression model effective for small datasets.

---

## 🧠 Workflow Steps

### 1️⃣ Data Collection & Preprocessing

* Data is sourced from Yahoo Finance using `yfinance` or `pandas_datareader`.
* Features like closing price, moving averages (MA), RSI, percent change, etc., are computed.
* Normalization is applied using MinMaxScaler or StandardScaler.

### 2️⃣ Model Training

#### 🔹 XGBoost:

* Trained on selected technical indicators.
* Optimized using `GridSearchCV` or `RandomSearchCV`.

#### 🔹 LSTM:

* Time series data is formatted into input sequences.
* Built using Keras or PyTorch with multi-layer LSTM architecture.
* `TimeSeriesGenerator` is used for batch training.

#### 🔹 SVR:

* Uses `sklearn.svm.SVR` with kernels such as `rbf`, `linear`, or `poly`.
* Effective for data with compact features.

### 3️⃣ Model Evaluation

* Metrics used: RMSE, MAE, and R² score.
* Visual plots compare actual vs. predicted prices.

---

## 🛠️ Setup Instructions

```bash
# Install required libraries
pip install yfinance scikit-learn xgboost keras pandas matplotlib
```

```python
# Load data from Yahoo Finance
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
```

```python
# Train XGBoost model
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
```

```python
# Train LSTM model
model = Sequential([
  LSTM(50, return_sequences=True),
  LSTM(50),
  Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---

## 📊 Model Performance

* **XGBoost**: Fast and powerful, but limited in long-term forecasting.
* **LSTM**: Good at capturing trends in complex time series.
* **SVR**: Performs well on short sequences with proper normalization.

<p align="center">
  <img src="https://user-images.githubusercontent.com/yourusername/yourrepo/prediction-results.png" width="600" />
</p>

---

## 🤝 Contributing

We welcome all contributions:

* Improve model accuracy
* Integrate additional technical indicators (MACD, Bollinger Bands, etc.)
* Deploy a real-time prediction web demo

---

## 📫 Contact

🌐 GitHub: [github.com/PVL-Linh](https://github.com/PVL-Linh)

