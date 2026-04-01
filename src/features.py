def create_features(df):
    df = df.copy()
    
    df["debt_to_income"] = df["debt"] / df["income"]
    df["is_young"] = (df["age"] < 25).astype(int)
    
    return df