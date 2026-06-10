import pandas as pd

df = pd.read_csv("data/cleaned_AAPL.csv")

target = "Close_forcast"

corr = df.corr(numeric_only=True)

target_corr = corr[target].abs()

important = target_corr.sort_values(
    ascending=False
)

print(important.head(20))

selected = important[
    important > 0.30
].index

print("\nSelected Features:")
print(selected)