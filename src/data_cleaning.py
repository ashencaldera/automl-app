import pandas as pd

def clean_data(df):
    df = df.drop_duplicates()
    df = df.ffill()
    return df