import pandas as pd

df = pd.read_csv("[all_countries]_[all_themes]_[monthly]_[vw_cap].csv")

countries = df["location"].unique()
factors = df["name"].unique()
date_range = pd.to_datetime(df["date"])

print(f"Rows: {len(df):,}")
print(f"Countries: {len(countries)}")
print(f"Factors: {len(factors)}")
print(f"Date range: {date_range.min().date()} to {date_range.max().date()}")
print(f"n_factors values: {sorted(df['n_factors'].unique())}")
print()
print("Countries:")
for c in sorted(countries):
    print(f"  {c}")
print()
print("Factors:")
for f in sorted(factors):
    print(f"  {f}")
print()
print("Return statistics:")
print(df["ret"].describe())
