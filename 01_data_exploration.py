import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AAPL.csv")

print("Shape:", df.shape)
print(df.info())
print(df.describe())

# Missing Values
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(18,12))
sns.heatmap(df.corr(numeric_only=True),
            cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()