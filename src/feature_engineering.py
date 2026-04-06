from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    df = df.copy()
    
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df