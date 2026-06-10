import pandas as pd

df = pd.read_csv("data/AAPL.csv")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Convert Date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

df.to_csv("data/cleaned_AAPL.csv",
          index=False)

print("Data Preparation Completed")