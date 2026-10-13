
import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:", df.isnull().sum().sum())

print("\nFinal grade summary:")
print(df["G3"].describe())
