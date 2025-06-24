import pandas as pd

df = pd.read_parquet(r'G:\Veraset\Visits\Veraset_Visits-119-UTC_TIMESTAMP-2019-09-19.snappy.parquet')
df['location_name'].value_counts()
df['top_category'].value_counts()
df['top_category'].isnull().sum()
df[df['location_name'] == 'home'].head().T
