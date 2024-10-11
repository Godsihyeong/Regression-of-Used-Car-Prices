import pandas as pd
import re
import numpy as np

def decode_engine(s: str):
    s = s.lower()
    
    engine_hp = float(re.search(r'(\d+(\.\d+)?)\s*hp', s).group(1)) if re.search(r'(\d+(\.\d+)?)\s*hp', s) else ''
    engine_cc = float(re.search(r'(\d+(\.\d+)?)\s*l', s).group(1)) if re.search(r'(\d+(\.\d+)?)\s*l', s) else ''
    engine_cyl = int(re.search(r'(\d+)\s*cylinder', s).group(1)) if re.search(r'(\d+)\s*cylinder', s) else ''
    
    turbo = bool(re.search(r'turbo', s))
    electric = bool(re.search(r'electric', s))

    return engine_hp, engine_cc, engine_cyl, turbo, electric

def clean_transmission(trans):
    trans = trans.lower()
    if any(keyword in trans for keyword in ['a/t', 'at', 'f', 'automatic', 'cvt', 'dual shift', 'overdrive']):
        return 'Automatic'
    elif any(keyword in trans for keyword in ['m/t', 'mt', 'manual', 'single-speed fixed gear']):
        return 'Manual'
    else:
        return 'Automatic'  # 특정하지 않은 값들은 'Unknown'으로 설정
    
def impute_data(series):
    na_mask = series.isna()     # searching null value -> return Boolean(True, False) 
    n_missing = na_mask.sum()   # calculating number of null value -> counting 'True'
    
    index = series.value_counts(normalize=True).index
    values = series.value_counts(normalize=True).values
    
    fill_values = np.random.choice(index, size = n_missing, p = values)
    series_copy = series.copy()
    series_copy.loc[na_mask] = fill_values
    
    return series_copy

def preprocess(df):
    print('start decoding features')
    df[['engine_hp', 'engine_cc', 'engine_cyl', 'engine_turbo', 'electric']] = df['engine'].apply(lambda x: pd.Series(decode_engine(x)))
    df['transmission_clean'] = df['transmission'].apply(clean_transmission)

    df['fuel_type'] = df['fuel_type'].replace('Plug-In Hybrid', 'Hybrid')

    df['fuel_type'] = df['fuel_type'].apply(lambda x : 'Gasoline' if pd.isna(x) else x)
    df['clean_title'] = df['clean_title'].apply(lambda x : 'No' if pd.isna(x) else x)
    df['accident'] = impute_data(df['accident'])

    df['engine_cc'] = df['engine_cc'].replace('', np.nan).astype(float)
    df['engine_cyl'] = df['engine_cyl'].replace('', np.nan).astype(float)
    df['engine_hp'] = df['engine_hp'].replace('', np.nan).astype(float)
    print('half')
    df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].mean())

    df.loc[df['electric'] == True, ['engine_cc', 'engine_cyl']] = df.loc[df['electric'] == True, ['engine_cc', 'engine_cyl']].fillna('Unknown')

    df['engine_cc'] = df['engine_cc'].fillna(df['engine_cc'].loc[df['engine_cc'] != 'Unknown'].astype(float).mean())
    df['engine_cyl'] = df['engine_cyl'].fillna(df['engine_cyl'].loc[df['engine_cyl'] != 'Unknown'].astype(float).mean())

    df['engine_cc'] = df['engine_cc'].apply(lambda x : 0 if x == 'Unknown' else x)
    df['engine_cyl'] = df['engine_cyl'].apply(lambda x : 0 if x == 'Unknown' else x)
    
    print('complete decoding features')
    
    return df

def extract_age_features(df):
    current_year = 2024

    df['Vehicle_Age'] = current_year - df['model_year']
    
    df['Mileage_per_Year'] = df['milage'] / df['Vehicle_Age']
    df['milage_with_age'] =  df.groupby('Vehicle_Age')['milage'].transform('mean')
    
    df['Mileage_per_Year_with_age'] =  df.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')
    
    return df


def extract_other_features(df):
    
    luxury_brands =  ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

    return df