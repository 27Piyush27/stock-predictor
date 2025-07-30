# stock-predictor
Stock Price Prediction Using Linear Regression
This project demonstrates a basic stock price prediction model using Linear Regression in Python. The example uses Apple Inc. (AAPL) historical stock data.

🛠️ Tech Stack / Libraries Used

yfinance: For fetching historical stock data.
numpy: For numerical operations.
scikit-learn: For machine learning modeling and evaluation.
matplotlib: For data visualization.
🔍 Project Objective

To predict stock prices using a simple linear regression model based on a time-indexed feature and evaluate the model’s performance using RMSE. A future stock price prediction is also made.

✅ Steps Covered

1. Import Libraries
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
2. Download Historical Stock Data
stock_data = yf.download('AAPL', start='2020-01-01', end='2025-07-30')
Checks if data was downloaded correctly.
Can be modified for any other stock ticker.
3. Prepare the Data
Create a time-based index as the only feature (X).
Use the 'Close' price as the target (y).
X = np.arange(len(stock_data)).reshape(-1, 1)
y = stock_data['Close'].values
4. Split Data into Training and Testing Sets
⚠️ In this example, train_test_split() is used, which randomly splits the data. For time series, consider chronological split instead.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
6. Make Predictions on the Test Set
y_pred = model.predict(X_test)
7. Evaluate the Model
Calculate RMSE (Root Mean Squared Error):
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
8. Visualize the Results
Plot actual data and model’s regression line:
plt.figure(figsize=(16, 8))
plt.title('Linear Regression: Stock Price Prediction vs. Actual Data')
...
plt.show()
9. Predict a Future Stock Price
Predict the stock price for the next day after the dataset ends:
future_price = model.predict([[len(stock_data)]])
print(f"Predicted future price: ${future_price[0]:.2f}")
📉 Example Output

RMSE: $X.XX (varies based on data)
Predicted future price: $XXX.XX
🚧 Limitations

Linear regression on a time index is over-simplified for stock prediction.
Does not account for market volatility, volume, indicators, or seasonality.
train_test_split() is not ideal for time-series—use chronological split instead.
Future prediction is extrapolation, which linear models handle poorly.
