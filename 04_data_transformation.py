import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/cleaned_AAPL.csv")

target = "Close_forcast"

X = df.drop(columns=[target])

X = X.select_dtypes(include='number')

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

scaled_df[target] = df[target]

scaled_df.to_csv(
    "data/scaled_AAPL.csv",
    index=False
)

print("Scaling Complete")