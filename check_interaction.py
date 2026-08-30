import pandas as pd

df = pd.read_csv("data/processed/sensitivity/studentlife_interaction_profiles.csv")
print(df.shape)
print(df.head(10))
print()
print("any NaN interaction_mean?", df["interaction_mean"].isna().any())
print()
print(df.groupby(["direction_a", "direction_b"])["interaction_mean"].mean())
