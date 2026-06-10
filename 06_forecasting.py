import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/scaled_AAPL.csv")

target = "Close_forcast"

X = df.drop(columns=[target])

y = df[target]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

future_data = X.tail(5)

forecast = model.predict(
    future_data
)

forecast_df = pd.DataFrame({
    "Day": range(
        1,
        len(forecast)+1
    ),
    "Forecast_Close": forecast
})

print(forecast_df)

forecast_df.to_csv(
    "outputs/forecast.csv",
    index=False
)